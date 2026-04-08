#include "launch_kernel.h"

#include <string>

#include "load_kernel.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

thread_local aclrtNotify g_allGatherBatchNotifies[kAllGatherBatchControlNotifyNum] = {nullptr};

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

    HCCL_INFO("Host launch done: rank=%u, commMode=%s, itemCount=%u",
        param.topoInfo.rank,
        ToCommModeString(param.commMode),
        param.itemCount);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch


