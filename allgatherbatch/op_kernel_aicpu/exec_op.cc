#include "exec_op.h"

#include "allgather_batch_small_count_executor.h"

namespace ops_hccl_allgatherbatch {

HcclResult ExecOp(const OpParam &param, AlgResourceCtx *resCtx, BatchCallProfiling &profiling)
{
    const uint64_t execStartUs = GetCurrentTimeUs();
    AllGatherBatchSmallCountExecutor executor(param, *resCtx, profiling);
    const HcclResult ret = executor.Orchestrate();
    profiling.execUs += (GetCurrentTimeUs() - execStartUs);
    return ret;
}

}  // namespace ops_hccl_allgatherbatch