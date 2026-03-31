#include "allgather_batch_small_count_executor.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

AllGatherBatchSmallCountExecutor::AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx)
    : param_(param), resCtx_(resCtx)
{
}

HcclResult AllGatherBatchSmallCountExecutor::Orchestrate()
{
    (void)param_;
    (void)resCtx_;
    HCCL_BATCH_INFO("phase 1 stub: executor not implemented yet");
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ops_hccl_allgatherbatch
