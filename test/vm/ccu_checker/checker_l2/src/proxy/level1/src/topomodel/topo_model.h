#ifndef SIM_TOPO_MODEL_H
#define SIM_TOPO_MODEL_H

#include "hccl_proxy_pub.h"
#include "hccl_common.h"
#include "hccl_rank_graph.h"
#include "hccl_sim_data_defs.h"
#include "hccl_sim_shm_manager.h"
#include "hccl_rankgraph.h"

namespace HcclProxy {
class TopoModel {
public:
    TopoModel() = default;
    std::vector<GraphRankInfo> rankGraphs_;
    ~TopoModel() = default;

    HcclResult Init();
    uint32_t GetRankSize() const;
    void GetNetLayers(uint32_t **netLayers, uint32_t *netLayerNum);
    HcclResult GetInstSizeByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t *rankNum);
    void GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink **linkList, uint32_t *listSize);
    void GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize);
    void GetInstTopoTypeByNetLayer(DevType devType, uint32_t netLayer, CommTopo *topoType);
    HcclResult GetInstRanksByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum);

private:
    void InitNetLayerInfo(uint32_t serverNum, uint32_t podNum);
    HcclResult InitLinkMap(uint32_t rankNum);

private:
    std::vector<uint32_t> netLayerList_;
    std::vector<uint32_t> allRankList_;
    std::map<uint32_t, std::vector<uint32_t>> serverId2RankList_;
    std::map<uint32_t, std::vector<uint32_t>> podId2RankList_;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<CommLink>> linkMap_;
    std::map<uint32_t, EndpointDesc> rankId2Endpoint_;

    std::vector<uint32_t> rankSizeGroup_;
    std::vector<uint32_t> serverSizeGroup_;
    std::vector<uint32_t> podSizeGroup_;
}; // TopoModel

};

#endif  // SIM_TOPO_MODEL_H