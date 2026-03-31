#include "allgather_batch_small_count_executor.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

AllGatherBatchSmallCountExecutor::AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx)
    : param_(param), resCtx_(resCtx)
{
}

HcclResult AllGatherBatchSmallCountExecutor::ValidateParam() const
{
    if (param_.itemCount == 0 || param_.itemCount > kAllGatherBatchMaxItems) {
        HCCL_ERROR("invalid itemCount=%u", param_.itemCount);
        return HCCL_E_PARA;
    }
    if (param_.resCtx == nullptr) {
        HCCL_ERROR("param.resCtx is null");
        return HCCL_E_PTR;
    }
    if (resCtx_.threadHandle == 0) {
        HCCL_ERROR("threadHandle is invalid");
        return HCCL_E_INTERNAL;
    }
    if (resCtx_.localBuffer.addr == nullptr || resCtx_.localBuffer.size == 0) {
        HCCL_ERROR("localBuffer is not ready");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::BuildFirstWindow(WindowRange &window) const
{
    // 阶段 3 先把窗口模型立起来：默认只有一个覆盖全部输入的初始窗口。
    window.startItemIdx = 0;
    window.startOffsetBytes = 0;
    window.endItemIdx = param_.itemCount - 1;
    window.endOffsetBytes = 0;
    window.packedBytes = param_.windowBytes;
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Orchestrate()
{
    HCCL_CHK_RET(ValidateParam());

    WindowRange firstWindow;
    HCCL_CHK_RET(BuildFirstWindow(firstWindow));

    // 阶段 3 到这里为止：执行器已经建立好最小的控制骨架。
    // 真正的 Pack/通信/Unpack 会在阶段 4/5 分别接入。
    HCCL_INFO("executor skeleton ready: itemCount=%u, windowBytes=%llu, localBufferSize=%llu",
        param_.itemCount,
        static_cast<unsigned long long>(firstWindow.packedBytes),
        static_cast<unsigned long long>(resCtx_.localBuffer.size));
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ops_hccl_allgatherbatch
