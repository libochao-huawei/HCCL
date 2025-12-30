/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_model.h"

namespace HcclSim {

constexpr uint32_t NetLayerL0 = 0;
constexpr uint32_t NetLayerL1 = 1;
constexpr uint32_t NetLayerL2 = 2;

TopoModel::TopoModel(const TopoMeta& topoMeta)
{
    uint32_t superpodId = 0;
    uint32_t serverId = 0;
    uint32_t rankId = 0;
    uint32_t devIpStart = 3232238090;
    for (auto& pod : topoMeta) {
        uint32_t rankNumInPod = 0;
        for (auto& server : pod) {
            uint32_t rankNumInServer = 0;
            for (auto& phyId : server) {
                rankId2ServerId_[rankId] = serverId;
                rankId2PodId_[rankId] = superpodId;
                serverId2RankList_[serverId].push_back(rankId);
                podId2RankList_[superpodId].push_back(rankId);
                allRankList_.push_back(rankId);

                // 初始化EndpointDesc
                EndpointDesc endpoint;
                CommAddr addr;
                addr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
                addr.id = htonl(devIpStart++);
                endpoint.commAddr = addr;
                endpoint.loc.device.devPhyId = phyId;
                endpoint.loc.device.superPodIdx = superpodId;
                endpoint.loc.device.serverIdx = serverId;
                rankId2Endpoint_[rankId] = endpoint;

                rankId++;
                rankNumInPod++;
                rankNumInServer++;
            }
            podServersGroup_.push_back(rankNumInServer);
            serverId++;
        }
        podRanksGroup_.push_back(rankNumInPod);
        superpodId++;
    }

    allRankNum_.push_back(rankId);
    InitNetLayerInfo(serverId, superpodId);
    InitLinkMap(rankId);
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

void TopoModel::InitLinkMap(uint32_t rankNum)
{
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
                if (rankId2ServerId_[srcRank] == rankId2ServerId_[dstRank]) {
                    protocol = CommProtocol::COMM_PROTOCOL_HCCS;
                } else {
                    protocol = CommProtocol::COMM_PROTOCOL_ROCE;
                }

                link.linkAttr.linkProtocol = protocol;
                linkMap_[rankPair].push_back(link);
            }
        }
    }
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

void TopoModel::GetInstSizeByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t *rankNum)
{
    auto serverId = rankId2ServerId_[curRank];
    auto podId = rankId2PodId_[curRank];
    if (netLayer == NetLayerL0) {
        *rankNum = serverId2RankList_[serverId].size();
    } else if (netLayer == NetLayerL1) {
        *rankNum = podId2RankList_[podId].size();
    } else {
        *rankNum = allRankList_.size();
    }
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
        *listSize = podServersGroup_.size();
        *instSizeList = podServersGroup_.data();
    } else if (netLayer == NetLayerL1) {
        *listSize = podRanksGroup_.size();
        *instSizeList = podRanksGroup_.data();
    } else {
        *listSize = allRankNum_.size();
        *instSizeList = allRankNum_.data();
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

void TopoModel::GetInstRanksByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum)
{
    auto serverId = rankId2ServerId_[curRank];
    auto podId = rankId2PodId_[curRank];
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
}

};