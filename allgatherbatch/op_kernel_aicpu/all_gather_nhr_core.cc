#include "all_gather_nhr_core.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

AllGatherNHRCore::AllGatherNHRCore(const OpParam &param, AlgResourceCtx &resCtx)
    : param_(param), resCtx_(resCtx)
{
}

HcclResult AllGatherNHRCore::RunAsync()
{
    (void)param_;
    (void)resCtx_;
    HCCL_INFO("phase 1 stub: NHR core not implemented yet");
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ops_hccl_allgatherbatch

