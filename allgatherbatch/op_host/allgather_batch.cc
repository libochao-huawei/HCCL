#include "allgather_batch.h"

#include "allgather_batch_op.h"
#include "log.h"

using ops_hccl_allgatherbatch::AllGatherBatchOp;

HcclResult HcclAllGatherBatch(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream)
{
    AllGatherBatchOp op;
    return op.Exec(items, itemCount, comm, stream);
}
