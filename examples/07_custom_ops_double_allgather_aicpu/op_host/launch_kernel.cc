#include "launch_kernel.h"
#include "load_kernel.h"

#include <string>`r`n#include <cstdlib>`r`n#include <cstring>

namespace ops_hccl_double_allgather {

thread_local aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];

HcclResult LaunchKernelWithAsc(DoubleAllGatherParam &param, aclrtStream stream)
{
    ACLCHECK(aclrtRecordNotify(g_notifies[0], stream));
    CHK_RET(LaunchKernelAsc(param, stream));
    ACLCHECK(aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchKernelWithAclrt(DoubleAllGatherParam &param, aclrtStream stream)
{
    CHK_RET(LoadAICPUKernel());
    ACLCHECK(aclrtRecordNotify(g_notifies[0], stream));

    constexpr const char *kernelName = "HcclLaunchDoubleAllGatherAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, kernelName, &funcHandle));
    ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
    aclrtParamHandle paraHandle;
    ACLCHECK(aclrtKernelArgsAppend(argsHandle, &param, sizeof(DoubleAllGatherParam), &paraHandle));
    ACLCHECK(aclrtKernelArgsFinalize(argsHandle));

    uint16_t waitTime = 27 * 68;
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = waitTime;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t numBlocks = 1;
    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr));
    ACLCHECK(aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(DoubleAllGatherParam &param, aclrtStream stream)
{
    char *kernelMode = getenv("HCCL_CUSTOM_KERNEL_LAUNCH_ASC");
    KernelLaunchMode mode = (kernelMode == nullptr || strcmp(kernelMode, "0") != 0) ? KERNEL_LAUNCH_ASC : KERNEL_LAUNCH_ACLRT;
    return (mode == KERNEL_LAUNCH_ASC) ? LaunchKernelWithAsc(param, stream) : LaunchKernelWithAclrt(param, stream);
}

}

