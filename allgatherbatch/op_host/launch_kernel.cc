#include "launch_kernel.h"

#include <map>
#include <mutex>
#include <string>

#include "load_kernel.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

thread_local aclrtNotify g_allGatherBatchNotifies[kAllGatherBatchControlNotifyNum] = {nullptr};

namespace {

struct HostProfilingRecord {
    BatchTopoInfo topoInfo {};
    BatchProfilingStats stats {};
};

std::mutex g_allGatherBatchProfilingMutex;
std::map<uint32_t, HostProfilingRecord> g_allGatherBatchProfilingRecords;

HcclResult CollectProfilingRecord(const OpParam &param)
{
    if (param.resCtx == nullptr) {
        return HCCL_SUCCESS;
    }

    AlgResourceCtx hostResCtx;
    std::memset(&hostResCtx, 0, sizeof(hostResCtx));
    ACLCHECK(aclrtMemcpy(
        &hostResCtx,
        sizeof(hostResCtx),
        param.resCtx,
        sizeof(hostResCtx),
        ACL_MEMCPY_DEVICE_TO_HOST));

    std::lock_guard<std::mutex> lock(g_allGatherBatchProfilingMutex);
    HostProfilingRecord &record = g_allGatherBatchProfilingRecords[param.topoInfo.rank];
    record.topoInfo = param.topoInfo;
    record.stats = hostResCtx.profiling;
    return HCCL_SUCCESS;
}

void DumpSingleProfilingRecord(uint32_t rank, const HostProfilingRecord &record)
{
    const BatchProfilingStats &stats = record.stats;
    const double callCount = (stats.callCount == 0) ? 1.0 : static_cast<double>(stats.callCount);
    const double avgWindowCount = static_cast<double>(stats.totalWindowCount) / callCount;
    const double avgKernelUs = static_cast<double>(stats.totalKernelUs) / callCount;
    const double avgExecUs = static_cast<double>(stats.totalExecUs) / callCount;
    const double avgPackUs = static_cast<double>(stats.totalPackUs) / callCount;
    const double avgHdStageUs = static_cast<double>(stats.totalHdStageUs) / callCount;
    const double avgUnpackUs = static_cast<double>(stats.totalUnpackUs) / callCount;

    HCCL_ERROR(
        "AGB_PROF rank=%u/%u calls=%llu avgWin=%.2f avg(us){kernel=%.2f exec=%.2f pack=%.2f hd=%.2f unpack=%.2f} last(us){kernel=%llu exec=%llu pack=%llu hd=%llu unpack=%llu} lastWin=%llu buf(B){local=%llu perRank=%llu maxWin=%llu input=%llu}",
        rank,
        record.topoInfo.rankSize,
        static_cast<unsigned long long>(stats.callCount),
        avgWindowCount,
        avgKernelUs,
        avgExecUs,
        avgPackUs,
        avgHdStageUs,
        avgUnpackUs,
        static_cast<unsigned long long>(stats.lastKernelUs),
        static_cast<unsigned long long>(stats.lastExecUs),
        static_cast<unsigned long long>(stats.lastPackUs),
        static_cast<unsigned long long>(stats.lastHdStageUs),
        static_cast<unsigned long long>(stats.lastUnpackUs),
        static_cast<unsigned long long>(stats.lastWindowCount),
        static_cast<unsigned long long>(stats.localBufferBytes),
        static_cast<unsigned long long>(stats.perRankCapacity),
        static_cast<unsigned long long>(stats.maxWindowBytes),
        static_cast<unsigned long long>(stats.totalInputBytes));
}

}  // namespace

HcclResult LaunchKernel(const OpParam &param, aclrtStream stream)
{
    HCCL_CHK_PTR(stream);

    ACLCHECK(aclrtRecordNotify(g_allGatherBatchNotifies[kAllGatherBatchControlNotifyStart], stream));

    aclrtFuncHandle funcHandle = nullptr;
    aclrtArgsHandle argsHandle = nullptr;
    aclrtParamHandle paramHandle = nullptr;
    ACLCHECK(aclrtBinaryGetFunction(g_allGatherBatchKernelHandle, kAllGatherBatchKernelName, &funcHandle));
    ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
    ACLCHECK(aclrtKernelArgsAppend(argsHandle, const_cast<OpParam *>(&param), sizeof(OpParam), &paramHandle));
    ACLCHECK(aclrtKernelArgsFinalize(argsHandle));

    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = 27 * 68;
    aclrtLaunchKernelCfg cfg;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t blockDim = 1;

    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, &cfg, argsHandle, nullptr));

    ACLCHECK(aclrtWaitAndResetNotify(
        g_allGatherBatchNotifies[kAllGatherBatchControlNotifyDone],
        stream,
        kAllGatherBatchCustomTimeoutMs));

    HCCL_CHK_RET(CollectProfilingRecord(param));

    HCCL_INFO("Host launch done: rank=%u, commMode=%s, itemCount=%u",
        param.topoInfo.rank,
        ToCommModeString(param.commMode),
        param.itemCount);
    return HCCL_SUCCESS;
}

HcclResult ResetHostProfiling()
{
    std::lock_guard<std::mutex> lock(g_allGatherBatchProfilingMutex);
    g_allGatherBatchProfilingRecords.clear();
    return HCCL_SUCCESS;
}

HcclResult DumpHostProfiling()
{
    std::lock_guard<std::mutex> lock(g_allGatherBatchProfilingMutex);
    if (g_allGatherBatchProfilingRecords.empty()) {
        HCCL_ERROR("AGB_PROF no-records");
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("AGB_PROF begin rankRecords=%u",
        static_cast<unsigned int>(g_allGatherBatchProfilingRecords.size()));
    for (const auto &entry : g_allGatherBatchProfilingRecords) {
        DumpSingleProfilingRecord(entry.first, entry.second);
    }
    HCCL_ERROR("AGB_PROF end");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch

extern "C" HcclResult HcclAllGatherBatchProfilingReset(void)
{
    return ops_hccl_allgatherbatch::ResetHostProfiling();
}

extern "C" HcclResult HcclAllGatherBatchProfilingDump(void)
{
    return ops_hccl_allgatherbatch::DumpHostProfiling();
}
