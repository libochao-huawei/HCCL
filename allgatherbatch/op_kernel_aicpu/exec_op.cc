#include "exec_op.h"

#include "allgather_batch_small_count_executor.h"

namespace ops_hccl_allgatherbatch {

namespace {

const char *ToCommModeString(BatchCommMode commMode)
{
    switch (commMode) {
        case BatchCommMode::kSingleServer:
            return "single-server";
        case BatchCommMode::kCrossServer:
            return "cross-server";
        default:
            return "unknown";
    }
}

}

HcclResult ExecOp(const OpParam &param, AlgResourceCtx *resCtx)
{
    HCCL_CHK_PTR(resCtx);
    HCCL_INFO("ExecOp dispatch: rank=%u, rankSize=%u, commMode=%s, channelCount=%u",
        param.topoInfo.rank,
        param.topoInfo.rankSize,
        ToCommModeString(param.commMode),
        resCtx->channelCount);

    // ExecOp 本身不承载算法细节，只负责做最薄的一层参数转发和入口日志。
    AllGatherBatchSmallCountExecutor executor(param, *resCtx);
    return executor.Orchestrate();
}

}  // namespace ops_hccl_allgatherbatch
