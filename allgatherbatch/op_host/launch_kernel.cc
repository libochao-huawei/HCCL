#include "launch_kernel.h"

#include "load_kernel.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

thread_local aclrtNotify g_allGatherBatchNotifies[kAllGatherBatchControlNotifyNum] = {nullptr};

HcclResult LaunchKernel(const OpParam &param, aclrtStream stream)
{
    (void)param;
    HCCL_CHK_PTR(stream);
    HCCL_CHK_RET(LoadAICPUKernel());

    // 这里先不进入真正的 notify + kernel launch 时序。
    // 原因是阶段 3 才会接好 Device 入口和参数协议，当前如果提前等待 notify，Host 侧会卡死。
    HCCL_WARNING("phase 2 host chain is ready, device launch path will be enabled in next phase");
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ops_hccl_allgatherbatch
