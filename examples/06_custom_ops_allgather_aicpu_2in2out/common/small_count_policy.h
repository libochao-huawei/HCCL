#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_SMALL_COUNT_POLICY_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_SMALL_COUNT_POLICY_H

#include "common.h"

namespace ops_hccl_allgather_2in2out {

struct SmallCountDecision {
    bool eligible = false;
    uint64_t loopMaxCount = 0;
    const char *reason = "uninitialized";
};

SmallCountDecision CheckFusedSmallCountEligible(
    const CommMeta &meta,
    uint64_t count0,
    uint64_t count1,
    HcclDataType dataType,
    uint64_t cclBufferSize);

} // namespace ops_hccl_allgather_2in2out

#endif
