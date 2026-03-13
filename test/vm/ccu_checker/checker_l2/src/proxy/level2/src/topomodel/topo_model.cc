#include "topo_model.h"

namespace HcclProxy {

constexpr uint32_t NetLayerL0 = 0;
constexpr uint32_t NetLayerL1 = 1;
constexpr uint32_t NetLayerL2 = 2;

HcclResult TopoModel::Init() {
    SHMManager::InitShm(false);
    ShmCommDomain* commDomain = SHMManager::FindShmObject<ShmCommDomain>(SHM_MODULE_COMM_DOMAIN);
    if (commDomain == nullptr) {
        std::cerr << "[TopoModel::Init] [ERROR] SHM obj not found"  << std::endl;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t superpodId = 0;
    uint32_t serverId = 0;
    uint32_t rankId = 0;
    uint32_t devIpStart = 3232238090;   // dev起始IP

    for (size_t i = 0; i < commDomain->rankNum; ++i) {
        allRankList_.push_back(i);
        superpodId = (commDomain->rankId2NpuPos)[i].field.podId;
        serverId = (commDomain->rankId2NpuPos)[i].field.serId;
        uint32_t phyDeviceId = (commDomain->rankId2NpuPos)[i].field.phyId;

        serverId2RankList_[serverId].push_back(i);
        podId2RankList_[superpodId].push_back(i);

        // 初始化EndpointDesc
        EndpointDesc endpoint;
        CommAddr addr;
        addr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
        addr.id = htonl(devIpStart++);
        endpoint.commAddr = addr;
        endpoint.loc.device.devPhyId = phyDeviceId;
        endpoint.loc.device.superPodIdx = superpodId;
        endpoint.loc.device.serverIdx = serverId;
        rankId2Endpoint_[i] = endpoint;
    }

    rankSizeGroup_.push_back(commDomain->rankNum);
    for (auto item : serverId2RankList_) {
        serverSizeGroup_.push_back((item.second).size());
    }
    for (auto item : podId2RankList_) {
        podSizeGroup_.push_back((item.second).size());
    }

    InitNetLayerInfo(serverSizeGroup_.size(), podSizeGroup_.size());
    auto ret = InitLinkMap(commDomain->rankNum);
    if (ret != HcclResult::HCCL_SUCCESS) {
        std::cerr << "[TopoModel::Init] [ERROR] InitLinkMap fail"  << std::endl;
        return ret;
    }
    return HcclResult::HCCL_SUCCESS;
}

void TopoModel::InitNetLayerInfo(uint32_t serverNum, uint32_t podNum)
{
    if (serverNum == 1) {
        netLayerList_ = {0};
    }

    if (serverNum > 1) {
        netLayerList_ = {0, 1};
    }

    if (podNum > 1) {
        netLayerList_ = {0, 1, 2};
    }
}

HcclResult TopoModel::InitLinkMap(uint32_t rankNum)
{
    SHMManager::InitShm(false);
    ShmCommDomain* commDomain = SHMManager::FindShmObject<ShmCommDomain>(SHM_MODULE_COMM_DOMAIN);
    if (commDomain == nullptr) {
        std::cerr << "[TopoModel::Init] [ERROR] SHM obj not found"  << std::endl;
        return HcclResult::HCCL_E_PARA;
    }

    for (uint32_t srcRank = 0; srcRank < rankNum; srcRank++) {
        for (uint32_t dstRank = 0; dstRank < rankNum; dstRank++) {
            if (srcRank == dstRank) {
                continue;
            }
            std::pair<uint32_t, uint32_t> rankPair = std::make_pair(srcRank, dstRank);
            for (uint32_t netLayer = 0; netLayer < 3; netLayer++) {
                CommLink link;
                link.srcEndpointDesc = rankId2Endpoint_[srcRank];
                link.dstEndpointDesc = rankId2Endpoint_[dstRank];
                CommProtocol protocol = CommProtocol::COMM_PROTOCOL_RESERVED;
                // 当前仅支持910B的互联
                uint32_t srcSerId = commDomain->rankId2NpuPos[srcRank].field.serId;
                uint32_t dstSerId = commDomain->rankId2NpuPos[dstRank].field.serId;
                if (srcSerId == dstSerId) {
                    protocol = CommProtocol::COMM_PROTOCOL_HCCS;
                } else {
                    protocol = CommProtocol::COMM_PROTOCOL_ROCE;
                }

                link.linkAttr.linkProtocol = protocol;
                linkMap_[rankPair].push_back(link);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

uint32_t TopoModel::GetRankSize() const
{
    return allRankList_.size();
}

void TopoModel::GetNetLayers(uint32_t **netLayers, uint32_t *netLayerNum)
{
    *netLayerNum = netLayerList_.size();
    *netLayers = netLayerList_.data();
}

HcclResult TopoModel::GetInstSizeByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t *rankNum)
{
    ShmNpuPos npuPos{};
    auto ret = GetNpuPosByRankId(curRank, &npuPos);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cerr << "[TopoModel::GetInstSizeByNetLayer] [ERROR] GetNpuPosByRankId fail"  << std::endl;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t serverId = npuPos.field.serId;
    uint32_t podId = npuPos.field.podId;
    if (netLayer == NetLayerL0) {
        *rankNum = serverId2RankList_[serverId].size();
    } else if (netLayer == NetLayerL1) {
        *rankNum = podId2RankList_[podId].size();
    } else {
        *rankNum = allRankList_.size();
    }
    return HcclResult::HCCL_SUCCESS;
}

void TopoModel::GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink **linkList, uint32_t *listSize)
{
    // L0也返回所有layer的连接
    std::pair<uint32_t, uint32_t> rankPair = std::make_pair(srcRank, dstRank);
    auto it = linkMap_.find(rankPair);
    if (it != linkMap_.end()) {
        *listSize = it->second.size();
        *linkList = it->second.data();
    } else {
        *listSize = 0;
        *linkList = nullptr;
    }
}

void TopoModel::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize)
{
    if (netLayer == NetLayerL0) {
        *listSize = serverSizeGroup_.size();
        *instSizeList = serverSizeGroup_.data();
    } else if (netLayer == NetLayerL1) {
        *listSize = podSizeGroup_.size();
        *instSizeList = podSizeGroup_.data();
    } else {
        *listSize = rankSizeGroup_.size();
        *instSizeList = rankSizeGroup_.data();
    }
}

void TopoModel::GetInstTopoTypeByNetLayer(DevType devType, uint32_t netLayer, CommTopo *topoType)
{
    if (netLayer == NetLayerL0) {
        if (devType == DevType::DEV_TYPE_910B) {
            *topoType = CommTopo::COMM_TOPO_1DMESH;
        } else if (devType == DevType::DEV_TYPE_910_93) {
            *topoType = CommTopo::COMM_TOPO_910_93;
        }
    } else if (netLayer == NetLayerL1) {
        *topoType = CommTopo::COMM_TOPO_CLOS;
    } else {
        *topoType = CommTopo::COMM_TOPO_CLOS;
    }
}

HcclResult TopoModel::GetInstRanksByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum)
{
    ShmNpuPos npuPos{};
    auto ret = GetNpuPosByRankId(curRank, &npuPos);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cerr << "[TopoModel::GetInstRanksByNetLayer] [ERROR] GetNpuPosByRankId fail"  << std::endl;
        return HcclResult::HCCL_E_PARA;
    }
    
    uint32_t serverId = npuPos.field.serId;
    uint32_t podId = npuPos.field.podId;
    if (netLayer == NetLayerL0) {
        *rankNum = serverId2RankList_[serverId].size();
        *ranks = serverId2RankList_[serverId].data();
    } else if (netLayer == NetLayerL1) {
        *rankNum = podId2RankList_[podId].size();
        *ranks = podId2RankList_[podId].data();
    } else {
        *rankNum = allRankList_.size();
        *ranks = allRankList_.data();
    }
    return HcclResult::HCCL_SUCCESS;
}

};