/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_multilevel_mesh1d.h"
#include "op_common.h"

namespace ops_hccl {
TopoMatchMultilevelMesh1D::TopoMatchMultilevelMesh1D()
    : TopoMatchBase()
{
}

TopoMatchMultilevelMesh1D::~TopoMatchMultilevelMesh1D()
{
}

HcclResult TopoMatchMultilevelMesh1D::TopoForLayer0(
    const HcclComm comm, uint32_t& layer0Size, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, uint32_t gcdInstSize) const
{
#ifndef AICPU_COMPILE
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));

    if (topoInstNum == NET_INST_NUM_1) {
        // mesh1d — 始终使用mesh1D，与TopoMatchMultilevel的layer0逻辑一致
        HCCL_INFO("[TopoMatchMultilevelMesh1d] layer0 topoInstNum [%d], Mesh 1D.", topoInstNum);
        uint32_t* ranks;
        uint32_t rankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[0], &ranks, &rankNum));

        if (gcdInstSize > 0 && gcdInstSize < rankNum) {
            // 非对称: 按GCD大小拆分pod为子组
            auto it = std::find(ranks, ranks + rankNum, myRank);
            CHK_PRT_RET(it == ranks + rankNum,
                HCCL_ERROR("[TopoMatchMultilevelMesh1d] [TopoForLayer0] myRank [%u] not found in ranks array", myRank),
                HcclResult::HCCL_E_INTERNAL);

            uint32_t myIdx = static_cast<uint32_t>(it - ranks);
            uint32_t groupId = myIdx / gcdInstSize;
            uint32_t startIdx = groupId * gcdInstSize;
            uint32_t endIdx = std::min(startIdx + gcdInstSize, rankNum);
            std::vector<uint32_t> rankVecLayer0(ranks + startIdx, ranks + endIdx);
            HCCL_DEBUG("[TopoMatchMultilevelMesh1d] [TopoForLayer0] Rank [%d], GCD subgroup: [%s]",
                myRank, PrintCArray<uint32_t>(rankVecLayer0.data(),
                static_cast<u32>(rankVecLayer0.size())).c_str());
            algHierarchyInfo.infos[0].push_back({rankVecLayer0});
            layer0Size = gcdInstSize;
        } else {
            // 对称: 整个pod作为一个组
            std::vector<uint32_t> rankVecLayer0(ranks, ranks + rankNum);
            algHierarchyInfo.infos[0].push_back({rankVecLayer0});
            layer0Size = rankVecLayer0.size();
        }
    } else if (topoInstNum == 0) {
        algHierarchyInfo.infos[0].push_back({myRank});
        layer0Size = 1;
    } else {
        // topoInstNum >= 2: 多实例拓扑，始终使用mesh1D
        // 将所有rank合并为一个扁平1D组，不区分x/y轴（与TopoMatch1D行为一致）
        HCCL_INFO("[TopoMatchMultilevelMesh1d] layer0 topoInstNum [%d], flattening to Mesh 1D.", topoInstNum);
        std::set<uint32_t> allRanksSet;
        for (uint32_t idx = 0; idx < topoInstNum; idx++) {
            uint32_t* ranks;
            uint32_t rankNum;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[idx], &ranks, &rankNum));
            for (uint32_t r = 0; r < rankNum; r++) {
                allRanksSet.insert(ranks[r]);
            }
        }
        std::vector<uint32_t> rankVecLayer0(allRanksSet.begin(), allRanksSet.end());
        algHierarchyInfo.infos[0].push_back({rankVecLayer0});
        layer0Size = rankVecLayer0.size();
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchMultilevelMesh1D::TopoForLayer1(
    const HcclComm comm, uint32_t netLayer, uint32_t& layer0Size, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
    HCCL_DEBUG("[TopoMatchMultilevelMesh1d::TopoForLayer1] layer0Size [%d]", layer0Size);
#ifndef AICPU_COMPILE
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum));
    CHK_PRT_RET(
        (topoInstNum != NET_INST_NUM_1),
        HCCL_ERROR("[TopoMatchMultilevelMesh1d::TopoForLayer1] layer1 topoInstNum [%d], Invalid topo.", topoInstNum),
        HcclResult::HCCL_E_PARA);

    uint32_t* ranks;
    uint32_t rankNum;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG("[TopoMatchMultilevelMesh1d::TopoForLayer1] Rank [%d], all [%u] ranks in layer1", myRank, rankNum);

    // 取出同序号卡，作为layer1的mesh1D组
    std::vector<uint32_t> rankVecLayer1WithSameIdx;
    for (uint32_t i = 0; i < rankNum; i++) {
        uint32_t rankId = ranks[i];
        if (myRank == rankId) {
            rankVecLayer1WithSameIdx.push_back(rankId);
            continue;
        }
        if (rankId % layer0Size != myRank % layer0Size) {
            continue;
        }
        CommLink *links;
        uint32_t linkNum = 0;
        HcclRankGraphGetLinks(comm, netLayer, myRank, rankId, &links, &linkNum);
        if (linkNum == 0) {
            continue;
        }
        rankVecLayer1WithSameIdx.push_back(rankId);
    }
    algHierarchyInfo.infos[1].push_back({rankVecLayer1WithSameIdx});
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchMultilevelMesh1D::TopoForLayer2(
    const HcclComm comm, uint32_t netLayer, uint32_t& layer0Size,
    uint32_t& layer1Size, const uint32_t myRank, AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
    HCCL_DEBUG("[TopoMatchMultilevelMesh1d::TopoForLayer2] layer1Size [%d]", layer1Size);
#ifndef AICPU_COMPILE
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum));
    CHK_PRT_RET(
        (topoInstNum != NET_INST_NUM_1),
        HCCL_ERROR("[TopoMatchMultilevelMesh1d::TopoForLayer2] layer2 topoInstNum [%d], Invalid topo.", topoInstNum),
        HcclResult::HCCL_E_PARA);

    uint32_t* ranks;
    uint32_t rankNum;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG("[TopoMatchMultilevelMesh1d::TopoForLayer2] Rank [%d], all [%u] ranks in layer2", myRank, rankNum);

    // 取出同子序号卡，作为layer2的mesh1D组
    // layer2的rank匹配条件: rankId在layer1组中与myRank具有相同的子序号
    std::vector<uint32_t> rankVecLayer2;
    for (uint32_t i = 0; i < rankNum; i++) {
        uint32_t rankId = ranks[i];
        if (myRank == rankId) {
            rankVecLayer2.push_back(rankId);
            continue;
        }
        // 跨超节点: 取同layer0序号且同layer1序号的rank
        // 简化条件: rankId与myRank在低两层中属于相同位置
        // 使用layer0Size*layer1Size作为分组粒度
        if (rankId % (layer0Size * layer1Size) != myRank % (layer0Size * layer1Size)) {
            continue;
        }
        CommLink *links;
        uint32_t linkNum = 0;
        HcclRankGraphGetLinks(comm, netLayer, myRank, rankId, &links, &linkNum);
        if (linkNum == 0) {
            continue;
        }
        rankVecLayer2.push_back(rankId);
    }
    algHierarchyInfo.infos[2].push_back({rankVecLayer2});
#endif
    return HcclResult::HCCL_SUCCESS;
}

bool TopoMatchMultilevelMesh1D::CheckVecElementAllSame(const uint32_t* instSizeList, uint32_t listSize) const
{
#ifndef AICPU_COMPILE
    if (listSize == 0) {
        return true;
    }
    uint32_t firstSize = instSizeList[0];
    for (uint32_t i = 1; i < listSize; i++) {
        if (firstSize != instSizeList[i]) {
            return false;
        }
    }
#endif
    return true;
}

uint32_t TopoMatchMultilevelMesh1D::GcdTwo(uint32_t a, uint32_t b) const
{
    while (b != 0) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

uint32_t TopoMatchMultilevelMesh1D::GcdOfInstSizeList(const uint32_t* instSizeList, uint32_t listSize) const
{
    if (listSize == 0) {
        return 1;
    }
    uint32_t result = instSizeList[0];
    for (uint32_t i = 1; i < listSize; i++) {
        result = GcdTwo(result, instSizeList[i]);
        if (result == 1) {
            return 1;
        }
    }
    return result;
}

HcclResult TopoMatchMultilevelMesh1D::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    // 支持1~3层拓扑
    CHK_PRT_RET(topoInfo->topoLevelNums == 0 || topoInfo->topoLevelNums > COMM_LAYER_SIZE_3,
        HCCL_ERROR("[TopoMatchMultilevelMesh1d] topoLevelNum[%u] is invalid, must be 1-3.",
            topoInfo->topoLevelNums),
        HCCL_E_INTERNAL);

    uint32_t myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));
    #ifdef MACRO_DEV_TYPE_NEW
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_950,
    #else
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_910_95,
    #endif
        HCCL_ERROR("[TopoMatchMultilevelMesh1d] Rank [%d], deviceType not supported yet.",
            myRank),
        HcclResult::HCCL_E_PARA);

    // 获取通信层数
    uint32_t *netLayers;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerNum));

    HCCL_DEBUG("[TopoMatchMultilevelMesh1d] Rank [%d], netLayers[%u][%s]",
                myRank, layerNum, PrintCArray<uint32_t>(netLayers, layerNum).c_str());

    // 初始化infos向量，大小取决于topo层级数
    uint32_t totalLevels = topoInfo->topoLevelNums;
    algHierarchyInfo.infos.resize(totalLevels);

    // 获取每个pod上rank数量以及pod数量
    uint32_t *instSizeList;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize));
    HCCL_INFO("[TopoMatchMultilevelMesh1d] Rank [%d], [%u] pods, ranksize on each pod:[%s]",
        myRank, listSize, PrintCArray<uint32_t>(instSizeList, listSize).c_str());
    bool isSymmetric = CheckVecElementAllSame(instSizeList, listSize);

    // 非对称仅支持 Mesh1D，提前校验
    if (!isSymmetric) {
        uint32_t *topoInsts;
        uint32_t topoInstNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));
        CHK_PRT_RET(topoInstNum != NET_INST_NUM_1,
            HCCL_ERROR("[TopoMatchMultilevelMesh1d][MatchTopo] Asymmetric mode only supports Mesh1D, "
                "but topoInstNum [%u]", topoInstNum),
            HcclResult::HCCL_E_NOT_SUPPORT);
    }

    // === Layer 0: pod内 Mesh1D ===
    uint32_t layer0Size = 0;
    if (!isSymmetric) {
        uint32_t gcdInstSize = GcdOfInstSizeList(instSizeList, listSize);
        HCCL_INFO("[TopoMatchMultilevelMesh1d][MatchTopo] Asymmetric mode, gcdInstSize [%u]", gcdInstSize);
        CHK_RET(TopoForLayer0(comm, layer0Size, myRank, algHierarchyInfo, gcdInstSize));
    } else {
        CHK_RET(TopoForLayer0(comm, layer0Size, myRank, algHierarchyInfo));
    }

    // === Layer 1: 跨pod Mesh1D (仅2层及以上) ===
    if (totalLevels >= COMM_LAYER_SIZE_2) {
        uint32_t netLayer = 1;
        bool hostDPUOnly = false;
        if ((CheckHostDPUOnly(comm, topoInfo, hostDPUOnly) == HcclResult::HCCL_SUCCESS) && hostDPUOnly) {
            netLayer = topoInfo->netLayerDetails.netLayers[topoInfo->netLayerDetails.netLayerNum - 1];
        }
        CHK_RET(TopoForLayer1(comm, netLayer, layer0Size, myRank, algHierarchyInfo));
    }

    // === Layer 2: 跨超节点 Mesh1D (仅3层) ===
    if (totalLevels >= COMM_LAYER_SIZE_3) {
        uint32_t layer1Size = algHierarchyInfo.infos[1][0].size();
        uint32_t netLayer = 2;
        // hostDPU场景使用最高层链路
        bool hostDPUOnly = false;
        if ((CheckHostDPUOnly(comm, topoInfo, hostDPUOnly) == HcclResult::HCCL_SUCCESS) && hostDPUOnly) {
            netLayer = topoInfo->netLayerDetails.netLayers[topoInfo->netLayerDetails.netLayerNum - 1];
        }
        CHK_RET(TopoForLayer2(comm, netLayer, layer0Size, layer1Size, myRank, algHierarchyInfo));
    }

    HCCL_INFO("[TopoMatchMultilevelMesh1d] Rank [%d], totalLevels [%u], layer0Size [%u] "
              "layer1Size [%u] layer2Size [%u]",
        myRank, totalLevels, layer0Size,
        (totalLevels >= 2) ? algHierarchyInfo.infos[1][0].size() : 0,
        (totalLevels >= 3) ? algHierarchyInfo.infos[2][0].size() : 0);
#endif
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl