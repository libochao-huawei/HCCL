#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_EXEC_OP_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_EXEC_OP_H

#include "common.h"

namespace ops_hccl_allgather_2in2out {

HcclResult ExecOp(OpParam &param, AlgResourceCtx *resCtx);

} // namespace ops_hccl_allgather_2in2out

#endif
