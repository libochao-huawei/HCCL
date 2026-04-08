#include "launch_kernel.h"

#include <string>

#include "load_kernel.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

thread_local aclrtNotify g_allGatherBatchNotifies[kAllGatherBatchControlNotifyNum] = {nullptr};

HcclResult LaunchKernel(const OpParam &param, aclrtStream stream)
{
    HCCL_CHK_PTR(stream);
    HCCL_CHK_RET(LoadAICPUKernel());

    if (g_allGatherBatchNotifies[kAllGatherBatchControlNotifyStart] == nullptr ||
        g_allGatherBatchNotifies[kAllGatherBatchControlNotifyDone] == nullptr) {
        HCCL_ERROR("host control notify is not ready");
        return HCCL_E_INTERNAL;
    }

    // Host stream 先发启动通知，Device 侧主线程收到后才真正进入执行器。
    ACLCHECK(aclrtRecordNotify(g_allGatherBatchNotifies[kAllGatherBatchControlNotifyStart], stream));

    // 把 Host 侧组织好的 OpParam 作为唯一 launch 入参带到 AICPU kernel 入口。
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

    // Device 入口先跑控制壳，再交给 ExecOp / Executor / HDStageCore。
    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, &cfg, argsHandle, nullptr));

    // Host 等待 Device 侧完成通知，形成完整的启动/结束闭环。
    ACLCHECK(aclrtWaitAndResetNotify(
        g_allGatherBatchNotifies[kAllGatherBatchControlNotifyDone],
        stream,
        kAllGatherBatchCustomTimeoutMs));

    HCCL_INFO("Host launch done: rank=%u, commMode=%s, itemCount=%u",
        param.topoInfo.rank,
        ToCommModeString(param.commMode),
        param.itemCount);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch


