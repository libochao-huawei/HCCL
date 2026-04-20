#include "launch_kernel.h"

#include <string>

#include "load_kernel.h"
#include "log.h"
#include "profiling.h"

namespace ops_hccl_allgatherbatch {

thread_local aclrtStream launchStream = nullptr;

HcclResult LaunchKernel(const OpParam &param, aclrtStream stream)
{
    HCCL_CHK_PTR(stream);
    if (launchStream == nullptr) {
        ACLCHECK(aclrtCreateStreamWithConfig(&launchStream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
    }
    HCCL_CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, param.aicpuThreadOnCpu, 0));

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
    const uint64_t beginTime = HcommGetProfilingSysCycleTime();

    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, blockDim, launchStream, &cfg, argsHandle, nullptr));
    
    CHK_PRT(HcommProfilingReportKernel(beginTime, kAllGatherBatchProfilingKernelName));

    HCCL_CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT));

    HCCL_INFO("Host launch done: rank=%u, commMode=%s, itemCount=%u",
        param.topoInfo.rank, ToCommModeString(param.commMode), param.itemCount);

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
