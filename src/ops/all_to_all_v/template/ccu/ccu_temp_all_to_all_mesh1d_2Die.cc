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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"

#include "ccu_temp_all_to_all_mesh1d_2Die.h"
#include "ccu_kernel_all_to_all_mesh2die.h"


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
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::ProcessLinkForProtocol(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound, const std::string& funcName)
{
    protocolFound = false;
    for (auto expectedProtocol : expectedProtocols) {
        for (u32 idx = 0; idx < linkList.size(); idx++) {
            if (linkList[idx].linkAttr.linkProtocol == expectedProtocol) {
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

HcclResult CcuTempAllToAllMesh1D2Die::CalcChannelRequest(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
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

    for (u32 rank: subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        size_t channelCountBefore = channels.size();
        uint32_t *netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

        // mesh 
        CommLink *linkListL1 = nullptr;
        u32 listSizeL1;
        CHK_RET(HcclRankGraphGetLinks(comm, 0, myRank, rank, &linkListL1, &listSizeL1));
        HCCL_INFO("CalcChannelRequest: netLayer[0], linkSize[%u]", listSizeL1);

        if (listSizeL1 == 0) {
            continue;
        }

        std::vector<CommLink> links0(linkListL1, linkListL1 + listSizeL1);
        bool protocolFound = false;
        CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links0, myRank, rank, 0, channels, protocolFound,
            std::string("[CalcChannelRequestMesh1D]")));
    
        CHK_PRT_RET(channels.size() == channelCountBefore,
            HCCL_ERROR("[CalcChannelRequestMesh1D] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                myRank, rank), HcclResult::HCCL_E_INTERNAL);
    }

    std::set<u32> connectRanks;
    u32 localRank = std::distance(subcommInfo[0].begin(), it);
    u32 localRankSize = subcommInfo[0].size();
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));

    for (u32 rankIdx: connectRanks) {
        size_t channelCountBefore = channels.size();
        uint32_t *netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

        // clos
        CommLink *linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, 1, myRank, subcommInfo[0][rankIdx], &linkList, &listSize));

        if (listSize == 0) {
            continue;
        }

        std::vector<CommLink> links(linkList, linkList + listSize);
        bool protocolFound = false;
        CHK_RET(ProcessLinkForProtocolNhr(comm, expectedProtocols, links, myRank, subcommInfo[0][rankIdx], 1, channels, protocolFound));

        CHK_PRT_RET(channels.size() == channelCountBefore,
            HCCL_ERROR("[CalcChannelRequestNhr] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                myRank, subcommInfo[0][rankIdx]), HcclResult::HCCL_E_INTERNAL);
    }

#endif
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    // 需要从流
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.push_back(1);

    //多少个kernel
    std::vector<HcclChannelDesc> channelDescs;
    // 要拿所有的channel，mesh和clos的。
    // mesh和clos的channel分开。通过layer去查，或者通过到某个对端的数量来判断，两个的就是clos。
    // 获取mesh的dieid。
    // 获取clos的dieid
    CHK_RET(CalcChannelRequest(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    HCCL_INFO("channelDescs size[%u]", channelDescs.size());

    uint32_t meshDieId = 0;
    CHK_RET(PartitionChannels(comm, channelDescs, meshDieId));
    resourceRequest.channels.emplace_back(channelDescs);
    HCCL_INFO("resourceRequest.channels[%d]",resourceRequest.channels.size());

    const uint32_t rankSize = subCommRanks_[0].size();
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);        // kernel数量
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        // 创建每个kernel的kernelArg，放入kernelInfo, 然后将kernelInfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelAllToAllMesh2Die>(arg);
        };
        const bool withMyRank = dieId == meshDieId ? true : false;
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllMesh2Die>(rankSize, myRank_, param, subCommRanks_,
            withMyRank, rankGroup_[dieId]);
        kernelInfo.kernelArg = kernelArg;
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][CalcRes] dieId=%u, channels=%llu, rankSize=%llu, ccuKernelInfos=%llu",
            dieId, channels_[dieId].size(), rankSize, resourceRequest.ccuKernelInfos.size());
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs, uint32_t &meshDieId)
{   // 目前channelDescs传入的是level0的
    // layer 0 -> mesh layer 1 -> clos 在mesh的时候查一下dieId，选择另外一个dieId的就是6口clos
    for (const auto &channel : channelDescs) {
        using DieIdType = uint32_t;
        const RankId remoteRank = channel.remoteRank;
        const uint32_t dieIdTypeSize = sizeof(DieIdType);
        EndpointDesc localEndpoint = channel.localEndpoint;
        HcclResult ret = HcclRankGraphGetEndpointInfo(comm, myRank_, &localEndpoint, ENDPOINT_ATTR_DIE_ID,
            dieIdTypeSize, static_cast<void*>(&meshDieId));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][CalcRes] Rank[%d] channel to remoteRank[%d], Failed to get dieId. "
                "errNo[0x%016llx]", myRank_, remoteRank, HCCL_ERROR_CODE(ret)),
            ret);
        CHK_PRT_RET(meshDieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][CalcRes] Rank[%d] channel to remoteRank[%d], dieId[%u] is invalid.",
                myRank_, remoteRank, meshDieId),
            HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempAllToAllMesh1D2Die][CalcRes] Rank[%d] channel to remoteRank[%d], insert to "
            "channels(dieId[%u]).", myRank_, channel.remoteRank, meshDieId);
        channels_[meshDieId].emplace_back(channel);
        rankGroup_[meshDieId].push_back(channel.remoteRank);
    }
    rankGroup_[0].push_back(myRank_);   // keep myRank_ at last, sync with kernel
    rankGroup_[1].push_back(myRank_);
    HCCL_INFO("channels[0].size[%u], channels[1].size[%u]", channels_[0].size(), channels_[1].size());
    uint32_t minChannels = std::min(channels_[0].size(), channels_[1].size());
    uint32_t maxChannels = std::max(channels_[0].size(), channels_[1].size());
    CHK_PRT_RET(minChannels + 1 != maxChannels,
        HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][CalcRes] Rank[%d], Unexpected channels size, "
            "die0 channels[%u], die1 channels[%u].", myRank_, channels_[0], channels_[1]),
        HcclResult::HCCL_E_PARA);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Run");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;

    const uint32_t rankSize = subCommRanks_[0].size();

    uint64_t inputAddr  = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t sliceSize        = templateDataParams.sliceSize;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t outBuffBaseOff =  buffInfo_.outBuffBaseOff;
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx], "
        "sendType[%d], recvType[%d]", myRank_, inputAddr, param.inputPtr, outputAddr, param.outputPtr,
        param.all2AllDataDes.sendType, param.all2AllDataDes.recvType);
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die][KernelRun] myRank_[%d], rankSize[%lu], inputAddr[%llu],"
              "outputAddr[%llu], sliceSize[%llu], outBuffBaseOff[%llu], inputSliceStride[%llu], outputSliceStride[%llu]",
               myRank_, rankSize, inputAddr, outputAddr, sliceSize, outBuffBaseOff, inputSliceStride, outputSliceStride);

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllMesh2Die>(
            inputAddr, outputAddr, token, sliceSize, inputSliceStride, outputSliceStride);
        void *taskArgPtr = static_cast<void *>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[dieId], templateResource.ccuKernels[dieId],
            taskArgPtr));
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][KernelRun] end. Rank[%d]", myRank_);
    HCCL_INFO("[CcuTempAllToAllMesh2Die] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
