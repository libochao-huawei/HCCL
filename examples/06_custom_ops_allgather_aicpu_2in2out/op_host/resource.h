#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_RESOURCE_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_RESOURCE_H

#include <vector>
#include "common.h"
#include "small_count_policy.h"

namespace ops_hccl_allgather_2in2out {

HcclResult BuildFusedOpParam(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    const SmallCountDecision &decision,
    const CommMeta &meta,
    const std::vector<uint32_t> &peers,
    HcclComm comm,
    OpParam &param);

} // namespace ops_hccl_allgather_2in2out

#endif
