/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_TOPO_MODEL_H
#define SIM_TOPO_MODEL_H

#include "hccl_sim_pub.h"
#include "hccl_common.h"
#include "hccl_rank_graph.h"

namespace HcclSim {
class TopoModel {
public:
    TopoModel() = delete;
    TopoModel(const TopoMeta& topoMeta);
    ~TopoModel() = default;
    uint32_t GetRankSize() const;
    void GetNetLayers(uint32_t **netLayers, uint32_t *netLayerNum);
    void GetInstSizeByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t *rankNum);
    void GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink **linkList, uint32_t *listSize);
    void GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize);
    void GetInstTopoTypeByNetLayer(DevType devType, uint32_t netLayer, CommTopo *topoType);
    void GetInstRanksByNetLayer(uint32_t curRank, uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum);

private:
    void InitNetLayerInfo(uint32_t serverNum, uint32_t podNum);
    void InitLinkMap(uint32_t rankNum);

private:
    std::vector<uint32_t> allRankList_;
    std::map<uint32_t, std::vector<uint32_t>> serverId2RankList_;
    std::map<uint32_t, std::vector<uint32_t>> podId2RankList_;
    std::vector<uint32_t> netLayerList_;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<CommLink>> linkMap_;
    std::map<uint32_t, EndpointDesc> rankId2Endpoint_;
    std::map<uint32_t, uint32_t> rankId2ServerId_;
    std::map<uint32_t, uint32_t> rankId2PodId_;

    std::vector<uint32_t> podServersGroup_;
    std::vector<uint32_t> podRanksGroup_;
    std::vector<uint32_t> allRankNum_;
}; // TopoModel

};

#endif  // SIM_TOPO_MODEL_H