#ifndef HCCL_ALLGATHERBATCH_PROFILING_H
#define HCCL_ALLGATHERBATCH_PROFILING_H

#include <cstdio>
#include <cstring>

#include "common.h"

namespace ops_hccl_allgatherbatch {

constexpr uint32_t kProfilingMaxLength = 128;
constexpr char kAllGatherBatchProfilingAlgType[] = "0-0-0";
constexpr char kAllGatherBatchProfilingKernelName[] = "allgatherbatch_aicpu_kernel";

#if defined(__GNUC__)
#define HCCL_ALLGATHERBATCH_WEAK __attribute__((weak))
#else
#define HCCL_ALLGATHERBATCH_WEAK
#endif

struct HcomProInfoTmp {
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[kProfilingMaxLength];
    char commName[kProfilingMaxLength];
    char algType[kProfilingMaxLength];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[kProfilingMaxLength];
};

extern "C" HCCL_ALLGATHERBATCH_WEAK uint64_t HcommGetProfilingSysCycleTime();
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingReportOp(HcomProInfoTmp profInfo);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingReportKernel(uint64_t beginTime, const char *profName);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingRegThread(HcomProInfoTmp profInfo, ThreadHandle *threads);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingUnRegThread(HcomProInfoTmp profInfo, ThreadHandle *threads);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingInit(ThreadHandle *threads, uint32_t threadNum);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingReportMainStreamAndLastTask(ThreadHandle thread);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingReportDeviceHcclOpInfo(HcomProInfoTmp profInfo);
extern "C" HCCL_ALLGATHERBATCH_WEAK HcclResult HcommProfilingEnd(ThreadHandle *threads, uint32_t threadNum);

inline bool IsHostProfilingEnabled()
{
    return (&HcommGetProfilingSysCycleTime != nullptr) &&
        (&HcommProfilingReportOp != nullptr) &&
        (&HcommProfilingReportKernel != nullptr);
}

inline bool IsHostThreadProfilingEnabled()
{
    return (&HcommProfilingRegThread != nullptr) &&
        (&HcommProfilingUnRegThread != nullptr);
}

inline bool IsDeviceProfilingEnabled()
{
    return (&HcommGetProfilingSysCycleTime != nullptr) &&
        (&HcommProfilingInit != nullptr) &&
        (&HcommProfilingReportMainStreamAndFirstTask != nullptr) &&
        (&HcommProfilingReportMainStreamAndLastTask != nullptr) &&
        (&HcommProfilingReportDeviceHcclOpInfo != nullptr) &&
        (&HcommProfilingEnd != nullptr);
}

inline bool IsProfilingEnabled()
{
    return IsHostProfilingEnabled() || IsDeviceProfilingEnabled();
}

inline uint32_t BuildProfilingThreadList(
    const AlgResourceCtx &resCtx,
    ThreadHandle *threads,
    uint32_t maxThreadNum)
{
    if (threads == nullptr || maxThreadNum == 0 || resCtx.mainThreadHandle == 0) {
        return 0;
    }

    uint32_t threadNum = 0;
    threads[threadNum++] = resCtx.mainThreadHandle;

    uint32_t workerCount = resCtx.lastTwoWorkerCount;
    if (workerCount > SubThreadNum) {
        workerCount = SubThreadNum;
    }
    for (uint32_t idx = 0; idx < workerCount && threadNum < maxThreadNum; ++idx) {
        if (resCtx.subThreadHandles[idx] != 0) {
            threads[threadNum++] = resCtx.subThreadHandles[idx];
        }
    }
    return threadNum;
}

inline void CopyProfilingString(char *dst, size_t dstSize, const char *src)
{
    if (dst == nullptr || dstSize == 0) {
        return;
    }
    dst[0] = '\0';
    if (src == nullptr) {
        return;
    }
    std::snprintf(dst, dstSize, "%s", src);
}

inline uint64_t AggregateProfilingDataCount(const OpParam &param, bool &dataTypeConsistent)
{
    dataTypeConsistent = true;
    if (param.itemCount == 0) {
        return 0;
    }

    const HcclDataType firstType = param.items[0].dataType;
    uint64_t totalCount = 0;
    for (uint32_t idx = 0; idx < param.itemCount; ++idx) {
        totalCount += param.items[idx].sendCount;
        if (param.items[idx].dataType != firstType) {
            dataTypeConsistent = false;
        }
    }
    return dataTypeConsistent ? totalCount : param.items[0].sendCount;
}

inline void FillProfilingInfo(
    HcomProInfoTmp &info,
    const OpParam &param,
    uint64_t beginTime,
    uint32_t slaveThreadNum)
{
    std::memset(&info, 0, sizeof(info));
    const bool hasItem = param.itemCount > 0;
    bool dataTypeConsistent = true;
    const uint64_t dataCount = AggregateProfilingDataCount(param, dataTypeConsistent);
    info.dataType = hasItem ? static_cast<uint8_t>(param.items[0].dataType) :
        static_cast<uint8_t>(HcclDataType::HCCL_DATA_TYPE_RESERVED);
    info.cmdType = static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    info.dataCount = dataCount;
    info.rankSize = param.topoInfo.rankSize;
    info.userRank = param.topoInfo.rank;
    info.blockDim = 1;
    info.beginTime = beginTime;
    info.root = 0;
    info.slaveThreadNum = slaveThreadNum;
    CopyProfilingString(info.tag, sizeof(info.tag), param.tag);
    CopyProfilingString(info.commName, sizeof(info.commName), param.commName);
    CopyProfilingString(info.algType, sizeof(info.algType), kAllGatherBatchProfilingAlgType);
    info.commNameLen = std::strlen(info.commName);
    info.algTypeLen = std::strlen(info.algType);
    (void)dataTypeConsistent;
}

}  // namespace ops_hccl_allgatherbatch

#endif  // HCCL_ALLGATHERBATCH_PROFILING_H
