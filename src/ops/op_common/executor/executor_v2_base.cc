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
    
HcclResult InsCollAlgBase::SetTempFastLaunchAddr(TemplateFastLaunchCtx &tempFastLaunchCtx, 
    void* inputPtr, void* outputPtr, const HcclMem &hcclBuff) const
{
    tempFastLaunchCtx.buffInfo.inputPtr = inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = outputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = hcclBuff;
    return HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::FastLaunch(const OpParam &param, const CcuFastLaunchCtx *resCtx)
{
    (void)param;
    (void)resCtx;
    HCCL_ERROR("[InsCollAlgBase] Unsupported interface of InsCollAlgBase::FastLaunch!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsCollAlgBase::CalAllLevelEndpointAttrBwCoeff(
    HcclComm comm, uint32_t rankId, uint32_t levelSize, std::vector<std::vector<EndpointAttrBwCoeff>> &endpointAttrBw)
{
    uint32_t *netLayers = nullptr; // 网络层次list
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));  // 获取layer总数和layerlist
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t netLayerId = netLayers[layerIdx];
        uint32_t *topoInsts = nullptr;
        uint32_t topoInstNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayerId, &topoInsts, &topoInstNum));  // 获取topoInstId
        // 同层可以有多个topoInstId，遍历获取
        for (uint32_t topoInsIdx = 0; topoInsIdx < topoInstNum; topoInsIdx++) {
            uint32_t topoInstId = topoInsts[topoInsIdx];
            uint32_t endPointNums = 0;
            CHK_RET(HcclRankGraphGetEndpointNum(comm, netLayerId, topoInstId, &endPointNums));  // 获取endPointNums，计算同层有多少节点
           EndpointDesc  *endPointDescs;
            CHK_RET(HcclRankGraphGetEndpointDesc(
                comm, netLayerId, topoInstId, &endPointNums, endPointDescs));  //根据Layer和topoInstId，拿到所有的Endpoint信息；返回vector(获取EndpointDesc)
            uint32_t infoLen = sizeof(EndpointAttrBwCoeff);
            EndpointAttrBwCoeff bwCoeff{};
            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, rankId, endPointDescs, ENDPOINT_ATTR_BW_COEFF, infoLen, &bwCoeff));  // 获取该维度的带宽
            endpointAttrBw.emplace_back(bwCoeff);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::FastLaunchSaveCtxTwoTemplate(const OpParam &param, const u32 threadNum,
    const u32 ccuKernelNum, const std::vector<ThreadHandle> &threads, const std::vector<u32> &ccuKernelNumList,
    const std::vector<std::vector<CcuKernelSubmitInfo>> &submitInfosList)
{
    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void *ctxPtr = nullptr;
    HCCL_INFO("[InsCollAlgBase][FastLaunchSaveCtxFourTemplate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsCollAlgBase][FastLaunchSaveCtxFourTemplate] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ThreadHandle *threadHandles = ccuFastLaunchCtx->GetThreadHandlePtr();
    for (u32 i = 0; i < threadNum; i++) {
        threadHandles[i] = threads[i];
    }

    // 3.1 ccu kernel数量
    for (u32 templateIdx = 0; templateIdx < ccuKernelNumList.size(); templateIdx++) {
        // 4次调用template.KernelRun下发kernel
        ccuFastLaunchCtx->ccuKernelNum[templateIdx] = ccuKernelNumList[templateIdx];
    }
    // 3.2 所有ccu kernel submitInfo打平放入数组
    constexpr u32 INTRA_0 = 0, INTER_1 = 1, INTER_0 = 2, INTRA_1 = 3;
    constexpr u32 INTRA = 0, INTER = 1;
    u32 kernelIdx = 0;
    CcuKernelSubmitInfo *kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    // 2个template实例，submitInfosList只有2份
    for (u32 i = 0; i < ccuKernelNumList[INTRA_0]; i++) {
        kernelSubmitInfos[kernelIdx++] = submitInfosList[INTRA][i];
    }
    for (u32 i = 0; i < ccuKernelNumList[INTER_1]; i++) {
        kernelSubmitInfos[kernelIdx++] = submitInfosList[INTER][i];
    }
    for (u32 i = ccuKernelNumList[INTER_1]; i < ccuKernelNumList[INTER_0] + ccuKernelNumList[INTER_1]; i++) {
        kernelSubmitInfos[kernelIdx++] = submitInfosList[INTER][i];
    }
    for (u32 i = ccuKernelNumList[INTRA_0]; i < ccuKernelNumList[INTRA_1] + ccuKernelNumList[INTRA_0]; i++) {
        kernelSubmitInfos[kernelIdx++] = submitInfosList[INTRA][i];
    }

    return HCCL_SUCCESS;
}

}