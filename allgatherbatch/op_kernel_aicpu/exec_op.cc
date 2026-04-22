#include "exec_op.h"

#include "allgather_batch_small_count_executor.h"

namespace ops_hccl_allgatherbatch {

HcclResult ExecOp(const OpParam &param, AlgResourceCtx *resCtx)
{
    AllGatherBatchSmallCountExecutor executor(param, *resCtx);
    return executor.Orchestrate();
}

}  // namespace ops_hccl_allgatherbatch
