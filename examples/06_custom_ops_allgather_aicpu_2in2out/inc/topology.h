#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_TOPOLOGY_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_TOPOLOGY_H

#include <vector>
#include "common.h"

namespace ops_hccl_allgather_2in2out {

TopologyType DetectTopologyType(const CommMeta &meta);
std::vector<uint32_t> BuildPeerOrder(uint32_t rankId, uint32_t rankSize, TopologyType topoType);

} // namespace ops_hccl_allgather_2in2out

#endif
