#ifndef HCCL_ALLGATHERBATCH_EXEC_OP_H
#define HCCL_ALLGATHERBATCH_EXEC_OP_H

#include "common.h"

namespace ops_hccl_allgatherbatch {

// Device 侧执行入口，负责把 launch 参数转交给执行器主控。
HcclResult ExecOp(const OpParam &param, AlgResourceCtx *resCtx);

}  // namespace ops_hccl_allgatherbatch

#endif
