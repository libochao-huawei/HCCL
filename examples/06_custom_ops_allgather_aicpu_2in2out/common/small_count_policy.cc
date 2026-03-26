#include "small_count_policy.h"

namespace ops_hccl_allgather_2in2out {

SmallCountDecision CheckFusedSmallCountEligible(
    const CommMeta &meta,
    uint64_t count0,
    uint64_t count1,
    HcclDataType dataType,
    uint64_t cclBufferSize)
{
    SmallCountDecision decision {};

    // 这一阶段先把基础边界卡严，避免把明显不支持的场景误判成 fused。
    if (meta.topologyType == TOPO_UNSUPPORTED) {
        decision.reason = "unsupported_topology";
        return decision;
    }

    if (meta.rankSize < 2) {
        decision.reason = "rank_size_lt_2";
        return decision;
    }

    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        decision.reason = "invalid_dtype";
        return decision;
    }

    const uint64_t unitSize = SIZE_TABLE[dataType];
    if (unitSize == 0) {
        decision.reason = "dtype_size_is_zero";
        return decision;
    }

    // 这里沿用设计文档中的 small-count 上界公式，先把最核心的 loop 容量算出来。
    uint64_t loopMaxCount = cclBufferSize / (unitSize * meta.rankSize);
    if (meta.rankSize % 4 == 0) {
        loopMaxCount *= 4;
    } else if (meta.rankSize % 2 == 0) {
        loopMaxCount *= 2;
    }

    decision.loopMaxCount = loopMaxCount;
    decision.eligible = (count0 <= loopMaxCount) && (count1 <= loopMaxCount);
    decision.reason = decision.eligible ? "ok" : "count_exceeds_loop_max";
    return decision;
}

} // namespace ops_hccl_allgather_2in2out
