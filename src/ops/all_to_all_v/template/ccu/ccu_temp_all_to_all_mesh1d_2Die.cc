/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "channel_request.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"

#include "ccu_temp_all_to_all_mesh1d_2Die.h"
#include "ccu_kernel_all_to_all_mesh2die.h"
#include "ccu_temp_all_to_all_mesh_1D.h"
#include "ccu_kernel_all_to_all_mesh1d.h"



namespace ops_hccl {
CcuTempAllToAllMesh1D2Die::CcuTempAllToAllMesh1D2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    for (u32 i = 0; i < subCommRanks_.size(); i++) {
        for (u32 j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_INFO("subCommRanks_[%u][%u]=%u", i, j, subCommRanks_[i][j]);
        }
    }

    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllToAllMesh1D2Die::~CcuTempAllToAllMesh1D2Die()
{
}

HcclResult CcuTempAllToAllMesh1D2Die::CreateChannelFromLink(HcclComm comm, u32 myRank, u32 rank, uint32_t netLayer, u32 idx,
    const CommLink& link, const std::string& funcName, std::vector<HcclChannelDesc>& channels)
{
    (void) comm;
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = rank;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    HCCL_DEBUG("%s local device phyId: %u, remote device phyId: %u.",
                funcName.c_str(), channelDesc.localEndpoint.loc.device.devPhyId,
                channelDesc.remoteEndpoint.loc.device.devPhyId);
    HCCL_INFO("%s Add channel request between %zu and %zu, netLayerIdx %u, "
              "linkListIdx %u, protocol %zu",
              funcName.c_str(), myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
    channels.push_back(channelDesc);
    HCCL_INFO("channels.size()=%llu",channels.size());
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::ProcessLinkForProtocol(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound, const std::string& funcName)
{
    HCCL_INFO("channels[].size()=%llu",channels.size());
    protocolFound = false;
    for (auto expectedProtocol : expectedProtocols) {
        for (u32 idx = 0; idx < linkList.size(); idx++) {
            HCCL_INFO("linkList.size()=%llu",linkList.size());
            if (linkList[idx].linkAttr.linkProtocol == expectedProtocol) {
                HCCL_INFO("llllllllllllllllllllll");
                CHK_RET(CreateChannelFromLink(comm, myRank, remoteRank, netLayer, idx, linkList[idx],
                    funcName, channels));
                protocolFound = true;
            }
        }
        if (protocolFound) {
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::ProcessLinkForProtocolNhr(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound)
{
    return ProcessLinkForProtocol(comm, expectedProtocols, linkList, myRank, remoteRank,
        netLayer, channels, protocolFound, std::string("[CalcLevel1ChannelRequestNhr]"));
}

HcclResult CcuTempAllToAllMesh1D2Die::CalcNHRChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks)
{
    (void)root;
    connectRanks.clear();
    if (rankSize == HCCL_RANK_SIZE_EQ_ONE) { // 只有一张卡时不需要建链
        HCCL_INFO("[CalcNHRChannelConnect] no need to create links, rankSize[%u].", rankSize);
        return HCCL_SUCCESS;
    }

    for (u32 delta = 1; delta < rankSize; delta <<= 1) {
        const u32 targetRankPos = static_cast<u32>(rank + delta) % rankSize;
        const u32 targetRankNeg = static_cast<u32>(rank + rankSize - delta) % rankSize;
        connectRanks.insert(targetRankPos);
        connectRanks.insert(targetRankNeg);
        HCCL_INFO("[CalcNHRChannelConnect]localRank[%u], rankPos[%u], rankNeg[%u]", rank, targetRankPos, targetRankNeg);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::CalcChannelRequest(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<std::vector<HcclChannelDesc>> &channels)
{
#ifndef AICPU_COMPILE
    (void) param;
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
                HcclResult::HCCL_E_PARA);

    u32 myRank = topoInfo->userRank;
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    uint32_t *netLayers, netLayerNum;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    channels.resize(netLayersVector.back() + 1);
    
    for (u32 rank: subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        HCCL_INFO("rank = %u",rank);

        for (auto netLayer : netLayersVector) {
            CommLink *linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
            if (listSize == 0) {
                continue;
            }
            HCCL_INFO("listSize = %u", listSize);
            std::vector<CommLink> links(linkList, linkList + listSize);
            bool protocolFound = false;
            CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links, myRank, rank, netLayer, channels[netLayer], protocolFound,
            std::string("[CalcChannelRequestMesh1D]")));
            HCCL_INFO("netLayer = %llu,channels[netLayer].size()= %llu,rank = %u",netLayer,channels[netLayer].size(),rank);
        }
        CHK_PRT_RET(channels.empty(),
            HCCL_ERROR("[CalcChannelRequestMesh1D] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                myRank, rank), HcclResult::HCCL_E_INTERNAL);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::RestoreChannelMap(const std::vector<std::vector<HcclChannelDesc>>& channelDescs,
                                std::map<u32, std::vector<std::vector<HcclChannelDesc>>>& rankIdToChannelDesc)
{

    for (size_t dieIdx = 0; dieIdx < channelDescs.size(); ++dieIdx) {
        const auto& channelList = channelDescs[dieIdx];
        for (const auto& channel : channelList) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelDesc[remoteRank].resize(2);
            rankIdToChannelDesc[remoteRank][dieIdx].push_back(channel);
            HCCL_INFO("remoteRank = %llu, dieIdx = %llu, rankIdToChannelDesc[remoteRank][dieIdx].size() = %llu", remoteRank, dieIdx, rankIdToChannelDesc[remoteRank][dieIdx].size());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    //多少个kernel
    std::vector<std::vector<HcclChannelDesc>> channelDescs;
    // 要拿所有的channel，mesh和clos的。
    // mesh和clos的channel分开。通过layer去查，或者通过到某个对端的数量来判断，两个的就是clos。
    // 获取mesh的dieid。
    // 获取clos的dieid
    CHK_RET(CalcChannelRequest(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    HCCL_INFO("channelDescs[0] size[%u], channelDescs[1] size[%u]", channelDescs[0].size(), channelDescs[1].size());

    uint32_t meshDieId = 0;
    CHK_RET(PartitionChannels(comm, channelDescs, meshDieId, rankIdToChannelDesc_));
    //resourceRequest.channels.emplace_back(channelDescs);
/*     for (auto& chGroup : channelDescs) {
        resourceRequest.channels.push_back(chGroup);
    } */
   std::vector<HcclChannelDesc> allChannels;
    for (auto& chGroup : channelDescs) {
        allChannels.insert(allChannels.end(), chGroup.begin(), chGroup.end());
    }

    // 最后只 push 一次！！！
    resourceRequest.channels.push_back(allChannels);
    HCCL_INFO("resourceRequest.channels[%d]",resourceRequest.channels.size());////////////1//////////0523:2

    const uint32_t rankSize = subCommRanks_[0].size();
    u32 kernelNum = (closChannels_[meshDieId].size() == 0) ? DIE_NUM: DIE_NUM + 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);        // kernel数量
    HCCL_INFO("closChannels_[meshDieId] = %llu", closChannels_[meshDieId].size());//////////15
    // 需要从流
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = (closChannels_[meshDieId].size() == 0) ? 1 : 2;//2+6需要2条从流，server需要1条从流
    resourceRequest.notifyNumPerThread.push_back(1);

    // 先下发mesh的kenrel
    CcuKernelInfo kernelInfoMesh;
    kernelInfoMesh.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllToAllMesh2Die>(arg);
    };
    auto kernelArgMesh = std::make_shared<CcuKernelArgAllToAllMesh2Die>(rankSize, myRank_, param, subCommRanks_,
        true, rankGroup_[meshDieId]);
    kernelInfoMesh.kernelArg = kernelArgMesh;
    kernelInfoMesh.channels = meshChannels_[meshDieId];
    resourceRequest.ccuKernelInfos.emplace_back(kernelInfoMesh);
    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][CalcRes] dieId=%u, channels=%llu, rankSize=%llu, ccuKernelInfos=%llu",
        meshDieId, meshChannels_[meshDieId].size(), rankSize, resourceRequest.ccuKernelInfos.size());

    // 下发clos的kenrel
    CcuKernelInfo kernelInfoClos;
    kernelInfoClos.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllToAllMesh2Die>(arg);
    };
    uint32_t closDieId = 1 - meshDieId;
    auto kernelArgClos = std::make_shared<CcuKernelArgAllToAllMesh2Die>(rankSize, myRank_, param, subCommRanks_,
        false, rankGroup_[closDieId]);
    kernelInfoClos.kernelArg = kernelArgClos;
    kernelInfoClos.channels = closChannels_[closDieId];
    resourceRequest.ccuKernelInfos.emplace_back(kernelInfoClos);
    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][CalcRes] dieId=%u, channels=%llu, rankSize=%llu, ccuKernelInfos=%llu",
        closDieId, closChannels_[closDieId].size(), rankSize, resourceRequest.ccuKernelInfos.size());

    //下发2port_clos的kernel
    if (closChannels_[meshDieId].size() == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    CcuKernelInfo kernelInfoClos2Port;
    kernelInfoClos2Port.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAlltoAllMesh1D>(arg);
    };
    auto kernelArgClos2Port = std::make_shared<CcuKernelArgAlltoAllMesh1D>(rankSize, myRank_, false, param, subCommRanks_);
    kernelInfoClos2Port.kernelArg = kernelArgClos2Port;
    kernelInfoClos2Port.channels = closChannels_[meshDieId];
    resourceRequest.ccuKernelInfos.emplace_back(kernelInfoClos2Port);
    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][CalcRes] dieId=%u, channels=%llu, rankSize=%llu, ccuKernelInfos=%llu",
        meshDieId, closChannels_[meshDieId].size(), rankSize, resourceRequest.ccuKernelInfos.size());
    
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::PartitionChannels(HcclComm comm, const std::vector<std::vector<HcclChannelDesc>> &channelDescs, uint32_t &meshDieId,
    std::map<u32, std::vector<std::vector<HcclChannelDesc>>>& rankIdToChannelDesc)
{   // 目前channelDescs传入的是level0的
    // layer 0 -> mesh layer 1 -> clos 在mesh的时候查一下dieId，选择另外一个dieId的就是6口clos
    for (auto& rankToChannels: rankIdToChannelDesc){
        u32 remoteRank = rankToChannels.first;
        std::vector<HcclChannelDesc>& meshChannel_list = rankToChannels.second[0];//mesh
        std::vector<HcclChannelDesc>& closChannel_list = rankToChannels.second[1];//clos

        using DieIdType = uint32_t;
        const uint32_t dieIdTypeSize = sizeof(DieIdType);
        //mesh链路
        if (!meshChannel_list.empty()){//该rank有mesh链路，取meshChannels_和meshDieId
            DieIdType dieId = 0;
            EndpointDesc localEndpoint = meshChannel_list.front().localEndpoint;
            HcclResult ret = HcclRankGraphGetEndpointInfo(comm, myRank_, &localEndpoint, ENDPOINT_ATTR_DIE_ID,
                dieIdTypeSize, static_cast<void*>(&dieId));
            meshChannels_[dieId].emplace_back(meshChannel_list.front());
            rankGroup_[dieId].push_back(meshChannel_list.front().remoteRank);
            meshDieId = dieId;
        }
        HCCL_INFO("meshChannels_[dieId].size() = %llu", meshChannels_[meshDieId].size());
        //clos链路
        if (!closChannel_list.empty()) {
            for (const auto &channel : closChannel_list) {
                DieIdType dieId = 0;
                EndpointDesc localEndpoint = channel.localEndpoint;
                HcclResult ret = HcclRankGraphGetEndpointInfo(comm, myRank_, &localEndpoint, ENDPOINT_ATTR_DIE_ID,
                    dieIdTypeSize, static_cast<void*>(&dieId));
                if (dieId == meshDieId) {
                    closChannels_[dieId].emplace_back(channel);//2port走mesh1d，需要所有channel
                    HCCL_INFO("closChannels_[dieId].size() = %llu", closChannels_[dieId].size());
                    HCCL_INFO("channel.remoteRank = %llu", channel.remoteRank);
                    HCCL_INFO("rankGroup_[dieId].size() = %llu", rankGroup_[dieId].size());
                } else if (remoteRank >= 8) {
                    closChannels_[dieId].emplace_back(channel);//6port走mesh2die，只需要跨框channel
                    rankGroup_[dieId].push_back(channel.remoteRank);
                    HCCL_INFO("closChannels_[dieId].size() = %llu", closChannels_[dieId].size());
                    HCCL_INFO("channel.remoteRank = %llu", channel.remoteRank);
                    HCCL_INFO("rankGroup_[dieId].size() = %llu", rankGroup_[dieId].size());
                }
            }
        }
         HCCL_INFO("closChannel_list.size() = %llu", closChannel_list.size());
    }
    HCCL_INFO("meshDieId = %llu", meshDieId);

    HCCL_INFO("closChannels_[0][%llu], closChannels_[1][%llu]", closChannels_[0].size(), closChannels_[1].size());

    rankGroup_[0].push_back(myRank_);   // keep myRank_ at last, sync with kernel
    rankGroup_[1].push_back(myRank_);
    
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::SplitDataFor2Dies(const OpParam& param,
                                                           const TemplateDataParams& templateDataParams,
                                                           uint64_t& sliceSizeMesh2die, uint64_t& sliceSizeMesh1d) const
{
    constexpr uint64_t MULTIPLIER = 4;
    uint64_t typeSize = DataTypeSizeGet(param.all2AllDataDes.recvType);
    uint64_t dataCount = (templateDataParams.sliceSize / typeSize);

    if (dataCount <= templateRankSize_ * MULTIPLIER) {   // 数据量极小，不划分die
        sliceSizeMesh2die = dataCount * typeSize;
        sliceSizeMesh1d = 0;
        return HcclResult::HCCL_SUCCESS;
    }
    u8 die0PortGroupSize = 6;
    u8 die1PortGroupSize = 2;

    sliceSizeMesh2die = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * typeSize;
    sliceSizeMesh1d = templateDataParams.sliceSize - sliceSizeMesh2die;
    HCCL_INFO("[CcuTempAllGatherNHR1DMem2Mem::SplitDataFor2Dies] sliceSizeMesh2die = %llu, sliceSizeMesh1d = %llu", sliceSizeMesh2die , sliceSizeMesh1d);
    
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Run");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    u32 kernelNum = templateResource.ccuKernels.size();
    const uint32_t rankSize = subCommRanks_[0].size();

    uint64_t inputAddr  = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    uint64_t sliceSizeMesh2die = 0;
    uint64_t sliceSizeMesh1d = 0;
    CHK_RET(GetToken(buffInfo_, token));

    if (kernelNum == DIE_NUM + 1) {
        SplitDataFor2Dies(param, templateDataParams, sliceSizeMesh2die, sliceSizeMesh1d);
    } else {
        sliceSizeMesh2die = templateDataParams.sliceSize;
    }

    // uint64_t inputSliceStride = templateDataParams.sdispls[1] * DATATYPE_SIZE_TABLE[param.all2AllDataDes.recvType] -  buffInfo_.inBuffBaseOff;
    uint64_t outputSliceStride = templateDataParams.sdispls[1] * DATATYPE_SIZE_TABLE[param.all2AllDataDes.recvType] -  buffInfo_.inBuffBaseOff;
    uint64_t inputSliceStride = outputSliceStride;
    uint64_t outBuffBaseOff =  buffInfo_.outBuffBaseOff;
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx], "
        "sendType[%d], recvType[%d]", myRank_, inputAddr, param.inputPtr, outputAddr, param.outputPtr,
        param.all2AllDataDes.sendType, param.all2AllDataDes.recvType);
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die][KernelRun] myRank_[%d], rankSize[%lu], inputAddr[%llu],"
              "outputAddr[%llu], sliceSizeMesh2die[%llu], outBuffBaseOff[%llu], inputSliceStride[%llu], outputSliceStride[%llu]",
               myRank_, rankSize, inputAddr, outputAddr, sliceSizeMesh2die, outBuffBaseOff, inputSliceStride, outputSliceStride);

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllMesh2Die>(
            inputAddr, outputAddr, token, sliceSizeMesh1d, inputSliceStride, outputSliceStride);
        void *taskArgPtr = static_cast<void *>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[dieId], templateResource.ccuKernels[dieId],
            taskArgPtr));
    }

    //判断是否有2port
    if (kernelNum == DIE_NUM + 1 && templateRankSize_ != 1) {
        //2port参数
        // 拿到input和output的首地址,和每片小数据的大小
        uint64_t srcStride = templateDataParams.outputSliceStride;
        uint64_t dstStride = templateDataParams.outputSliceStride;
        uint64_t srcOffset = 0;
        uint64_t dstOffset = myRank_ * dstStride;

        HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Run Init: myRank_[%d],  inputAddr[%llu],"\
            "outputAddr[%llu], sliceSizeMesh1d[%llu], srcOffset[%llu], dstOffset[%llu]",
            myRank_, inputAddr, outputAddr, sliceSizeMesh1d, srcOffset, dstOffset);
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAlltoAllMesh1D>(
            inputAddr, outputAddr, sliceSizeMesh1d, token, srcOffset, dstOffset, srcStride);

        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[DIE_NUM], templateResource.ccuKernels[DIE_NUM], taskArgPtr));
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
