#include <cstring>
#include <string>
#include "load_kernel.h"
#include "launch_kernel.h"

namespace ops_hccl_allgather_2in2out {

thread_local aclrtNotify g_notifies[kControlNotifyNum] = {nullptr};

HcclResult EnsureHostControlNotifiesCreated()
{
    // 这两个 notify 负责把 Host stream 和 AICPU kernel 的执行边界串起来。
    for (uint32_t idx = 0; idx < kControlNotifyNum; ++idx) {
        if (g_notifies[idx] == nullptr) {
            ACLCHECK(aclrtCreateNotify(&g_notifies[idx], ACL_NOTIFY_DEFAULT));
        }
    }
    return HCCL_SUCCESS;
}

namespace {

HcclResult LaunchKernelWithAclrt(OpParam &param, aclrtStream stream)
{
    CHK_RET(LoadAICPUKernel());
    CHK_RET(EnsureHostControlNotifiesCreated());

    // Host stream 先发启动信号，让 AICPU 后续可以在正确时点开始执行。
    ACLCHECK(aclrtRecordNotify(g_notifies[0], stream));

    std::string kernelName = "HcclLaunchAllGather2In2OutAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle));
    ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));

    aclrtParamHandle paraHandle;
    ACLCHECK(aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle));
    ACLCHECK(aclrtKernelArgsFinalize(argsHandle));

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    std::memset(&cfg, 0, sizeof(cfg));
    std::memset(&attr, 0, sizeof(attr));
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = 27 * 68;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;

    constexpr uint32_t numBlocks = 1;
    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr));

    // Host 侧等待 AICPU 回 completion notify，确保后续依赖不会越过本次 kernel。
    ACLCHECK(aclrtWaitAndResetNotify(g_notifies[1], stream, kCustomTimeout));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult LaunchKernel(OpParam &param, aclrtStream stream)
{
    // 阶段 3/4 先统一走 ACLRT launch 方式，后续如有需要再补 ASC 路径。
    return LaunchKernelWithAclrt(param, stream);
}

} // namespace ops_hccl_allgather_2in2out
