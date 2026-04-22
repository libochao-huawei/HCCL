#ifndef HCCL_ALLGATHERBATCH_EXEC_OP_H
#define HCCL_ALLGATHERBATCH_EXEC_OP_H

#include "common.h"

namespace ops_hccl_allgatherbatch {

HcclResult ExecOp(const OpParam &param, AlgResourceCtx *resCtx);

}  // namespace ops_hccl_allgatherbatch

#endif
