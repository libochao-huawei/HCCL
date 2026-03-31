#include "all_gather_hd_stage_core.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

AllGatherHDStageCore::AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx)
    : param_(param), resCtx_(resCtx)
{
}

HcclResult AllGatherHDStageCore::RunAsync()
{
    (void)param_;
    (void)resCtx_;
    HCCL_INFO("phase 1 stub: HD stage core not implemented yet");
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ops_hccl_allgatherbatch

