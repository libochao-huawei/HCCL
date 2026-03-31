#include "common.h"
#include "exec_op.h"

extern "C" HcclResult HcclAllGatherBatchAicpuKernel(
    ops_hccl_allgatherbatch::OpParam *param)
{
    using namespace ops_hccl_allgatherbatch;
    HCCL_BATCH_CHK_PTR(param);
    HCCL_BATCH_CHK_PTR(param->resCtx);
    return ExecOp(*param, param->resCtx);
}
