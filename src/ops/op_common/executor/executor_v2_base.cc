/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_v2_base.h"
#include "adapter_error_manager_pub.h"

namespace ops_hccl {

InsCollAlgBase::InsCollAlgBase()
{
}

InsCollAlgBase::~InsCollAlgBase()
{
}

std::string InsCollAlgBase::Describe() const
{
    std::string s = "111";
    return s;
}

HcclResult InsCollAlgBase::RestoreChannelMap(const AlgResourceCtxSerializable &resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>> &rankIdToChannelInfo) const
{
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo = resCtx.algHierarchyInfo;
    rankIdToChannelInfo.resize(algHierarchyInfo.infos.size());
    for (u32 level = 0; level < algHierarchyInfo.infos.size(); level++) {
        for (auto &channel: resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[level][remoteRank].push_back(channel);
        }
        // 不需要再resize内层的map，因为map会自动管理元素
    }
    return HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::CalAllLevelEndpointAttrBwCoeff(
    HcclComm comm, u32 rankId, u32 levelSize, std::vector<std::vector<EndpointAttrBwCoeff>> &endpointAttrBw)
{
    // uint32_t myRank;
    // CHK_RET(HcclGetRankId(comm, &myRank));//获取userrank，也可通过executor传进去
    u32 *netLayers = nullptr;  // 网络层次list
    u32 netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));  // 获取layer总数和layerlist
    for (u32 layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        u32 netLayerId = netLayers[layerIdx];
        u32 *topoInsts = nullptr;
        u32 topoInstNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayerId, &topoInsts, &topoInstNum));  // 获取topoInstId
        // 同层可以有多个topoInstId，遍历获取
        for (u32 topoInsIdx = 0; topoInsIdx < topoInstNum; topoInsIdx++) {
            u32 topoInstId = topoInsts[topoInsIdx];
            u32 endPointNums = 0;
            CHK_RET(HcclRankGraphGetEndpointNum(
                comm, netLayerId, topoInstId, &endPointNums));  // 获取endPointNums，计算同层有多少节点
            EndpointDesc *endPointDescs;
            // 根据Layer和topoInstId，拿到所有的Endpoint信息；返回vector(获取EndpointDesc)
            CHK_RET(HcclRankGraphGetEndpointDesc(comm, netLayerId, topoInstId, &endPointNums, endPointDescs));
            u32 infoLen = sizeof(EndpointAttrBwCoeff);
            EndpointAttrBwCoeff bwCoeff{};
            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, rankId, endPointDescs, ENDPOINT_ATTR_BW_COEFF, infoLen, &bwCoeff));  // 获取该维度的带宽
            endpointAttrBw.emplace_back(bwCoeff);
        }
    }
    return HCCL_SUCCESS;
}
}