#include "allgather_batch.h"

#include "allgather_batch_op.h"
#include "log.h"

using ops_hccl_allgatherbatch::AllGatherBatchOp;

HcclResult HcclAllGatherBatch(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream)
{
    // 对外入口先只保留轻量日志，详细拓扑/模式信息交给 Host 主链路在拿到 topo 后再打印。
    HCCL_INFO("HcclAllGatherBatch invoked: itemCount=%u", itemCount);
    AllGatherBatchOp op;
    return op.Exec(items, itemCount, comm, stream);
}
