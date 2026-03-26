#include "topology.h"

namespace ops_hccl_allgather_2in2out {

TopologyType DetectTopologyType(const CommMeta &meta)
{
    // 阶段 2 先使用保守启发式：
    // 1. rankSize < 2 视为当前样例无需进入 collective 路径；
    // 2. rankSize <= 8 先按单机多卡理解；
    // 3. 更大的规模先按超节点内多机占位，后续阶段再接真实拓扑查询。
    if (meta.rankSize < 2) {
        return TOPO_UNSUPPORTED;
    }
    if (meta.rankSize <= 8) {
        return TOPO_SINGLE_NODE;
    }
    return TOPO_INTRA_SUPERPOD_MULTI_NODE;
}

std::vector<uint32_t> BuildPeerOrder(uint32_t rankId, uint32_t rankSize, TopologyType topoType)
{
    std::vector<uint32_t> peers;
    if (topoType == TOPO_UNSUPPORTED || rankSize < 2) {
        return peers;
    }

    peers.reserve(rankSize - 1);
    // 先使用最容易理解的轮转顺序：从自己后一个 rank 开始依次遍历。
    for (uint32_t step = 1; step < rankSize; ++step) {
        peers.push_back((rankId + step) % rankSize);
    }
    return peers;
}

} // namespace ops_hccl_allgather_2in2out
