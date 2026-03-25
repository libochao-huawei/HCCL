#ifndef OPS_HCCL_DOUBLE_ALLGATHER_EXEC_H
#define OPS_HCCL_DOUBLE_ALLGATHER_EXEC_H

#include "common.h"

namespace ops_hccl_double_allgather {

HcclResult ExecDoubleAllGather(DoubleAllGatherParam &param, AlgResourceCtx *resCtx);

}

#endif
