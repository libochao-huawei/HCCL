#ifndef HCCL_ALLGATHERBATCH_PROFILING_H
#define HCCL_ALLGATHERBATCH_PROFILING_H

#include <cstdio>
#include <cstring>

#include "common.h"

namespace ops_hccl_allgatherbatch {

constexpr char kAllGatherBatchProfilingAlgType[] = "0-0-0";
constexpr char kAllGatherBatchProfilingKernelName[] = "allgatherbatch_aicpu_kernel";

#ifndef MAX_LENGTH
#define MAX_LENGTH 128
#endif

typedef struct HcomProInfoTmp {
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
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
} HcomProInfoTmp;

extern "C" bool HcommIsProfilingSupported();

#if defined(HOST_COMPILE)
extern "C" uint64_t (*hcommGetProfilingSysCycleTimePtr)();
extern "C" HcclResult (*hcommProfilingReportOpPtr)(HcomProInfoTmp);
extern "C" HcclResult (*hcommProfilingReportKernelPtr)(uint64_t, const char *);
extern "C" HcclResult (*hcommProfilingRegThreadPtr)(HcomProInfoTmp, ThreadHandle *);
extern "C" HcclResult (*hcommProfilingInitPtr)(ThreadHandle *, uint32_t);
extern "C" HcclResult (*hcommProfilingReportMainStreamAndFirstTaskPtr)(ThreadHandle);
extern "C" HcclResult (*hcommProfilingReportMainStreamAndLastTaskPtr)(ThreadHandle);
extern "C" HcclResult (*hcommProfilingReportDeviceHcclOpInfoPtr)(HcomProInfoTmp);
extern "C" HcclResult (*hcommProfilingEndPtr)(ThreadHandle *, uint32_t);

#define HcommGetProfilingSysCycleTime (*hcommGetProfilingSysCycleTimePtr)
#define HcommProfilingReportOp (*hcommProfilingReportOpPtr)
#define HcommProfilingReportKernel (*hcommProfilingReportKernelPtr)
#define HcommProfilingRegThread (*hcommProfilingRegThreadPtr)
#define HcommProfilingInit (*hcommProfilingInitPtr)
#define HcommProfilingReportMainStreamAndFirstTask (*hcommProfilingReportMainStreamAndFirstTaskPtr)
#define HcommProfilingReportMainStreamAndLastTask (*hcommProfilingReportMainStreamAndLastTaskPtr)
#define HcommProfilingReportDeviceHcclOpInfo (*hcommProfilingReportDeviceHcclOpInfoPtr)
#define HcommProfilingEnd (*hcommProfilingEndPtr)

extern "C" bool HcommIsSupportHcommGetProfilingSysCycleTime(void);
extern "C" bool HcommIsSupportHcommProfilingReportOp(void);
extern "C" bool HcommIsSupportHcommProfilingReportKernel(void);
extern "C" bool HcommIsSupportHcommProfilingRegThread(void);
extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void);
extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void);
extern "C" bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void);
extern "C" bool HcommIsSupportHcommProfilingInit(void);
extern "C" bool HcommIsSupportHcommProfilingEnd(void);
#else
extern "C" HcclResult HcommProfilingInit(ThreadHandle *, uint32_t);
extern "C" HcclResult HcommProfilingReportMainStreamAndFirstTask(ThreadHandle);
extern "C" HcclResult HcommProfilingReportMainStreamAndLastTask(ThreadHandle);
extern "C" HcclResult HcommProfilingReportDeviceHcclOpInfo(HcomProInfoTmp);
extern "C" HcclResult HcommProfilingEnd(ThreadHandle *, uint32_t);

extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void);
extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void);
extern "C" bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void);
extern "C" bool HcommIsSupportHcommProfilingInit(void);
extern "C" bool HcommIsSupportHcommProfilingEnd(void);
#endif

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
    info.isCapture = false;
    info.isAiv = false;
    (void)dataTypeConsistent;
}

}  // namespace ops_hccl_allgatherbatch

#endif  // HCCL_ALLGATHERBATCH_PROFILING_H
