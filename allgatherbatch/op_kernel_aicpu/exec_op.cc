#include "exec_op.h"

#include "allgather_batch_small_count_executor.h"

namespace ops_hccl_allgatherbatch {

HcclResult ExecOp(const OpParam &param, AlgResourceCtx *resCtx)
{
    HCCL_CHK_PTR(resCtx);
    HCCL_INFO("ExecOp dispatch: rank=%u, rankSize=%u, commMode=%s, channelCount=%u, crossServerChannels=%u, perRankCapacity=%llu, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param.topoInfo.rank,
        param.topoInfo.rankSize,
        ToCommModeString(param.commMode),
        resCtx->channelCount,
        CountCrossServerChannels(param.topoInfo, *resCtx),
        static_cast<unsigned long long>(GetPerRankWindowCapacity(param, *resCtx)),
        CountChannelsByProtocol(*resCtx, COMM_PROTOCOL_HCCS),
        CountChannelsByProtocol(*resCtx, COMM_PROTOCOL_ROCE),
        CountChannelsByProtocol(*resCtx, COMM_PROTOCOL_PCIE),
        CountChannelsByProtocol(*resCtx, COMM_PROTOCOL_SIO));

    // ExecOp 本身不承载算法细节，只负责做最薄的一层参数转发和入口日志。
    AllGatherBatchSmallCountExecutor executor(param, *resCtx);
    return executor.Orchestrate();
}

}  // namespace ops_hccl_allgatherbatch

