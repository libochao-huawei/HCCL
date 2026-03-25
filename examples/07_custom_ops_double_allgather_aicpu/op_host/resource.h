#ifndef OPS_HCCL_DOUBLE_ALLGATHER_RESOURCE_H
#define OPS_HCCL_DOUBLE_ALLGATHER_RESOURCE_H

#include "common.h"

namespace ops_hccl_double_allgather {

HcclResult PrepareResources(HcclComm comm, DoubleAllGatherParam &param, aclrtStream stream);

}

#endif
