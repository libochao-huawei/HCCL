#include "allgather_batch_op.h"

#include <cstdio>

#include "launch_kernel.h"
#include "load_kernel.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

HcclResult AllGatherBatchOp::Exec(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream)
{
    HCCL_BATCH_CHK_RET(Validate(items, itemCount, comm, stream));

    OpParam param;
    HCCL_BATCH_CHK_RET(PrepareOpParam(items, itemCount, comm, param));

    AlgResourceCtx resCtx;
    HCCL_BATCH_CHK_RET(GetAlgRes(comm, param.topoInfo, resCtx));
    param.resCtx = &resCtx;

    HCCL_BATCH_CHK_RET(LoadAndLaunch(param, stream));
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::Validate(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream) const
{
    HCCL_BATCH_CHK_PTR(items);
    HCCL_BATCH_CHK_PTR(comm);
    HCCL_BATCH_CHK_PTR(stream);
    if (itemCount == 0 || itemCount > kAllGatherBatchMaxItems) {
        HCCL_BATCH_ERROR("invalid itemCount=%u", itemCount);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::PrepareTopoInfo(HcclComm comm, BatchTopoInfo &topoInfo) const
{
    HCCL_BATCH_CHK_RET(HcclGetRankId(comm, &topoInfo.rank));
    HCCL_BATCH_CHK_RET(HcclGetRankSize(comm, &topoInfo.rankSize));
    topoInfo.serverCount = 0;
    topoInfo.serverIdx = 0;
    topoInfo.superPodIdx = 0;
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::PrepareOpParam(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, OpParam &param) const
{
    (void)items;
    std::snprintf(param.tag, sizeof(param.tag), "%s", "allgatherbatch");
    HCCL_BATCH_CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_BATCH_CHK_RET(PrepareTopoInfo(comm, param.topoInfo));
    param.itemCount = itemCount;
    param.appendedItemBytes = static_cast<uint64_t>(itemCount) * sizeof(BatchItemParam);
    param.windowBytes = 0;
    param.totalInputBytes = 0;
    param.totalOutputBytes = 0;
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::GetAlgRes(
    HcclComm comm, const BatchTopoInfo &topoInfo, AlgResourceCtx &resCtx) const
{
    (void)topoInfo;
    void *localBuffer = nullptr;
    HCCL_BATCH_CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU, 1, 0, &resCtx.threadHandle));
    HCCL_BATCH_CHK_RET(HcclGetHcclBuffer(comm, &localBuffer, &resCtx.localBuffer.size));
    resCtx.localBuffer.addr = localBuffer;
    resCtx.channelCount = 0;
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::LoadAndLaunch(const OpParam &param, aclrtStream stream) const
{
    HCCL_BATCH_CHK_RET(LoadAICPUKernel());
    HCCL_BATCH_CHK_RET(LaunchKernel(param, stream));
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
