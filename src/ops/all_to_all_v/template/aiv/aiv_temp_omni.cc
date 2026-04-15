/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv/aiv_temp_omni.h"

#include <algorithm>
#include <set>
#include <vector>

#include "alg_data_trans_wrapper.h"
#include "adapter_acl.h"
#include "hccl_aiv_utils.h"
#include "op_common.h"
#include "channel.h"
#include "all_to_all_v/template/ccu/kernel/ccu_kernel_omni.h"
#include "all_to_all_v/template/aiv/kernel/aiv_kernel_omni.h"

namespace ops_hccl {
constexpr u32 MAX_NUM_BLOCKS_OMNI = 1;

AivTempOmni::AivTempOmni(const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[COMM_LEVEL0].size();
}

AivTempOmni::~AivTempOmni()
{
}

HcclResult AivTempOmni::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    (void) comm;
    (void) param;
    (void) topoInfo;
    (void) resourceRequest;
    HCCL_ERROR("[AivTempOmni][CalcRes] xml info is required for omni resource calculation.");
    return HCCL_E_INTERNAL;
}

HcclResult AivTempOmni::CalcChannelRequestOmni(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    const std::vector<std::map<u32, OmniChannelInfo>>& mapchannelInfo, std::vector<HcclChannelDesc>& channels)
{
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
        HCCL_ERROR("[AivTempOmni][CalcChannelRequestOmni] Rank [%d] is not in commInfo.", topoInfo->userRank),
        HcclResult::HCCL_E_PARA);

    const u32 myRank = topoInfo->userRank;
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    std::set<u64> connectedRanks;
    for (u32 layer = 0; layer < mapchannelInfo.size(); layer++) {
        for (const auto& pair : mapchannelInfo[layer]) {
            const u64 remoteRank = pair.first;
            if (connectedRanks.count(remoteRank) > 0) {
                continue;
            }

            uint32_t *netLayers = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
            std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

            bool channelCreated = false;
            for (auto netLayer : netLayersVector) {
                CommLink *linkList = nullptr;
                u32 listSize = 0;
                CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, static_cast<u32>(remoteRank), &linkList, &listSize));

                for (u32 idx = 0; idx < listSize; idx++) {
                    bool protocolMatched = false;
                    for (auto expectedProtocol : expectedProtocols) {
                        if (linkList[idx].linkAttr.linkProtocol == expectedProtocol) {
                            protocolMatched = true;
                            break;
                        }
                    }
                    if (!protocolMatched) {
                        continue;
                    }

                    HcclChannelDesc channelDesc;
                    HcclChannelDescInit(&channelDesc, 1);
                    channelDesc.remoteRank = static_cast<u32>(remoteRank);
                    channelDesc.localEndpoint.protocol = linkList[idx].srcEndpointDesc.protocol;
                    channelDesc.localEndpoint.commAddr = linkList[idx].srcEndpointDesc.commAddr;
                    channelDesc.localEndpoint.loc = linkList[idx].srcEndpointDesc.loc;
                    channelDesc.remoteEndpoint.protocol = linkList[idx].dstEndpointDesc.protocol;
                    channelDesc.remoteEndpoint.commAddr = linkList[idx].dstEndpointDesc.commAddr;
                    channelDesc.remoteEndpoint.loc = linkList[idx].dstEndpointDesc.loc;
                    channelDesc.channelProtocol = linkList[idx].linkAttr.linkProtocol;
                    channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
                    channels.push_back(channelDesc);
                    channelCreated = true;
                    break;
                }
                if (channelCreated) {
                    break;
                }
            }

            if (channelCreated) {
                connectedRanks.insert(remoteRank);
            } else {
                HCCL_WARNING("[AivTempOmni][CalcChannelRequestOmni] No matching link found for myRank[%u] remoteRank[%llu].",
                    myRank, remoteRank);
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AivTempOmni::BuildInstructionBuffer(HcclComm comm, const OpParam& param, const XmlInfo& xmlInfo)
{
    const uint64_t infoNum = xmlInfo.vecSendRecvInfo.size();
    const uint64_t ctxSize = sizeof(AivOmniInfoHeader) + infoNum * sizeof(AivOmniSendRecvInfo);
    std::vector<u8> hostBuffer(ctxSize, 0);
    auto *header = reinterpret_cast<AivOmniInfoHeader *>(hostBuffer.data());
    auto *infos = reinterpret_cast<AivOmniSendRecvInfo *>(hostBuffer.data() + sizeof(AivOmniInfoHeader));
    header->infoNum = infoNum;

    for (u64 idx = 0; idx < infoNum; idx++) {
        const auto &srcInfo = xmlInfo.vecSendRecvInfo[idx];
        CHK_PRT_RET(srcInfo.srcSliceInfo.size() > AIV_OMNI_MAX_SLICE_CNT ||
            srcInfo.dstSliceInfo.size() > AIV_OMNI_MAX_SLICE_CNT,
            HCCL_ERROR("[AivTempOmni][BuildInstructionBuffer] slice count exceeds max supported size[%u].",
                AIV_OMNI_MAX_SLICE_CNT), HCCL_E_NOT_SUPPORT);
        infos[idx].opType = static_cast<u32>(srcInfo.optype);
        infos[idx].inputDataType = static_cast<u32>(srcInfo.inputDataType);
        infos[idx].outputDataType = static_cast<u32>(srcInfo.outputDataType);
        infos[idx].reduceType = static_cast<u32>(srcInfo.reduceType);
        infos[idx].srcSliceNum = srcInfo.srcSliceInfo.size();
        infos[idx].dstSliceNum = srcInfo.dstSliceInfo.size();
        infos[idx].sliceNum = srcInfo.sliceNum;
        infos[idx].linkType = srcInfo.netlayerId;
        infos[idx].threadIdx = srcInfo.threadIdx;
        for (u32 sliceIdx = 0; sliceIdx < infos[idx].srcSliceNum; sliceIdx++) {
            infos[idx].srcSliceInfo[sliceIdx].sliceType =
                static_cast<uint64_t>(srcInfo.srcSliceInfo[sliceIdx].sliceType);
            infos[idx].srcSliceInfo[sliceIdx].sliceIdx = srcInfo.srcSliceInfo[sliceIdx].sliceIdx;
            infos[idx].srcSliceInfo[sliceIdx].remoteRank = srcInfo.srcSliceInfo[sliceIdx].remoteRank;
        }
        for (u32 sliceIdx = 0; sliceIdx < infos[idx].dstSliceNum; sliceIdx++) {
            infos[idx].dstSliceInfo[sliceIdx].sliceType =
                static_cast<uint64_t>(srcInfo.dstSliceInfo[sliceIdx].sliceType);
            infos[idx].dstSliceInfo[sliceIdx].sliceIdx = srcInfo.dstSliceInfo[sliceIdx].sliceIdx;
            infos[idx].dstSliceInfo[sliceIdx].remoteRank = srcInfo.dstSliceInfo[sliceIdx].remoteRank;
        }
    }

    void *ctx = nullptr;
    uint64_t currentSize = 0;
    if (HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &currentSize) != HCCL_SUCCESS) {
        CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, ctxSize, &ctx));
        currentSize = ctxSize;
    }
    CHK_PRT_RET(currentSize < ctxSize,
        HCCL_ERROR("[AivTempOmni][BuildInstructionBuffer] device ctx size[%llu] is smaller than required[%llu].",
            currentSize, ctxSize), HCCL_E_INTERNAL);
    CHK_RET(haclrtMemcpy(ctx, ctxSize, hostBuffer.data(), ctxSize, ACL_MEMCPY_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult AivTempOmni::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest, const XmlInfo& xmlInfo)
{
    resourceRequest.slaveThreadNum = xmlInfo.resInfo.slaveThreadNum;
    resourceRequest.notifyNumOnMainThread = xmlInfo.resInfo.notifyNumOnMainThread;
    resourceRequest.notifyNumPerThread.assign(xmlInfo.resInfo.notifyNumPerThread, 1);
    numBlocks_ = std::max<u32>(1, xmlInfo.resInfo.blockNumAiv);
    numBlocks_ = std::min<u32>(numBlocks_, MAX_NUM_BLOCKS);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestOmni(comm, param, topoInfo, subCommRanks_, xmlInfo.resInfo.mapchannelInfo, channelDescs));
    resourceRequest.channels.emplace_back(channelDescs);
    CHK_RET(BuildInstructionBuffer(comm, param, xmlInfo));
    return HCCL_SUCCESS;
}

HcclResult AivTempOmni::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void) dataSize;
    (void) numBlocksLimit;
    numBlocks_ = MAX_NUM_BLOCKS_OMNI;
    numBlocks = numBlocks_;
    return HCCL_SUCCESS;
}

u64 AivTempOmni::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void) inBufferType;
    (void) outBufferType;
    return 1;
}

HcclResult AivTempOmni::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    IncSliceId();
    dataType_ = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) ? param.all2AllVDataDes.sendType : param.DataDes.dataType;

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    CHK_RET(HcclEngineCtxGet(param.hcclComm, param.algTag, param.engine, &ctx, &ctxSize));

    AivOpArgs omniArgs;
    omniArgs.cmdType = param.opType;
    omniArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    omniArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    omniArgs.rank = static_cast<u32>(myRank_);
    omniArgs.rankSize = tempRankSize_;
    omniArgs.count = tempAlgParams.sliceSize / SIZE_TABLE[dataType_];
    omniArgs.dataType = dataType_;
    omniArgs.op = param.reduceType;
    omniArgs.root = root_;
    omniArgs.sliceId = static_cast<u32>(sliceId_);
    omniArgs.buffersIn = templateResource.aivCommInfoPtr;
    omniArgs.stream = param.stream;
    omniArgs.isOpBase = (param.opMode == OpMode::OPBASE);
    omniArgs.xRankSize = subCommRanks_[0].size();
    omniArgs.yRankSize = 0;
    omniArgs.zRankSize = 0;
    omniArgs.extraArgs.omniInfoAddr = reinterpret_cast<u64>(ctx);
    omniArgs.extraArgs.omniInfoSize = ctxSize;

    for (u32 i = 0; i < subCommRanks_[0].size(); i++) {
        omniArgs.topo_[i] = subCommRanks_[0][i];
    }
    if (subCommRanks_.size() > 1) {
        omniArgs.yRankSize = subCommRanks_[1].size();
        for (u32 i = 0; i < subCommRanks_[1].size(); i++) {
            omniArgs.topo_[TOPO_LEN_Y_OFFSET + i] = subCommRanks_[1][i];
        }
    }
    if (subCommRanks_.size() == MAX_DIM_NUM) {
        omniArgs.zRankSize = subCommRanks_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < subCommRanks_[MAX_DIM_NUM - 1].size(); i++) {
            omniArgs.topo_[TOPO_LEN_Z_OFFSET + i] = subCommRanks_[MAX_DIM_NUM - 1][i];
        }
    }

    CHK_RET(CalNumBlocks(omniArgs.numBlocks, tempAlgParams.inputSliceStride, MAX_NUM_BLOCKS_OMNI));
    omniArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    omniArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    omniArgs.repeatNum = tempAlgParams.repeatNum;
    omniArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    omniArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    ExecuteKernelLaunch(omniArgs);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
