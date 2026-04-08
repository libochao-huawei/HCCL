/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "template_utils.h"

#include "ccu_temp_omni.h"

namespace ops_hccl {

constexpr u32 DIE_NUM = 2;
constexpr u32 UDIE0 = 0;
constexpr u32 UDIE1 = 1;

CcuTempOmni::CcuTempOmni(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

CcuTempOmni::~CcuTempOmni()
{
}

HcclResult CcuTempOmni::CreateChannelFromLink(HcclComm comm, u32 myRank, u32 rank, uint32_t netLayer, u32 idx,
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

HcclResult CcuTempOmni::ProcessLinkForProtocol(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound, const std::string& funcName)
{
    protocolFound = false;
    HCCL_INFO("ProcessLinkForProtocol expectedProtocols size %u", expectedProtocols.size());
    for (auto expectedProtocol : expectedProtocols) {
        for (u32 idx = 0; idx < linkList.size(); idx++) {
            HCCL_INFO("linkProtocol %u, expectedProtocol %u", linkList[idx].linkAttr.linkProtocol, expectedProtocol);
            if (linkList[idx].linkAttr.linkProtocol == expectedProtocol) {
                CHK_RET(CreateChannelFromLink(comm, myRank, remoteRank, netLayer, idx, linkList[idx],
                    funcName, channels));
                protocolFound = true;
                break;
            }
        }
        if (protocolFound) {
            break;
        }
    }
    return HCCL_SUCCESS;
}

// HcclResult CcuTempOmni::CalcChannelRequestOmni(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
//     const std::vector<std::vector<u32>>& subcommInfo, const std::vector<std::map<u32, OmniChannelInfo>>& mapchannelInfo, 
//     std::vector<HcclChannelDesc> &channels)
// {
//     (void) param;
//     channels.clear();
//     auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
//     CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
//                 HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
//                 HcclResult::HCCL_E_PARA);

//     u32 myRank = topoInfo->userRank;
//     HCCL_DEBUG("mapchannelInfo size %u", mapchannelInfo.size());
//     for (u32 i = 0; i < mapchannelInfo.size(); i++) { // neylayer size
//         for (auto& pair : mapchannelInfo[i]) {
//             uint64_t remoteRank = pair.first;
//             uint64_t channelId = pair.second.channelId;

//             CommLink *linkList = nullptr;
//             u32 listSize;
//             CHK_RET(HcclRankGraphGetLinks(comm, i, myRank, remoteRank, &linkList, &listSize));
//             if (listSize == 0) {
//                 HCCL_DEBUG("%u to %u link size is 0", myRank, remoteRank);
//                 continue;
//             }

//             HCCL_DEBUG("%u to %u link size is %u", myRank, remoteRank, listSize);

//             std::vector<CommLink> links(linkList, linkList + listSize);
//             bool protocolFound = false;
//             std::vector<CommProtocol> expectedProtocols;
//             expectedProtocols.push_back(pair.second.channelProtocol);

//             CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links, myRank, remoteRank, i, channels, protocolFound,
//                 std::string("[CalcChannelRequestMesh1D]")));
//         }   

//         HCCL_DEBUG("channels size is %u", channels.size());     
//     }

//     return HCCL_SUCCESS;
// }

HcclResult CcuTempOmni::CalcChannelRequestOmni(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, const std::vector<std::map<u32, OmniChannelInfo>>& mapchannelInfo, 
    std::vector<HcclChannelDesc> &channels)
{
    (void) param;
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
                HcclResult::HCCL_E_PARA);

    u32 myRank = topoInfo->userRank;
    HCCL_DEBUG("mapchannelInfo size %u", mapchannelInfo.size());
    for (u32 i = 0; i < mapchannelInfo.size(); i++) { // neylayer size
        for (auto& pair : mapchannelInfo[i]) {
            uint64_t remoteRank = pair.first;
            CommProtocol protocol = pair.second.channelProtocol;
            // uint64_t channelId = pair.second.channelId;

            CommLink *linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, i, myRank, remoteRank, &linkList, &listSize));

            HCCL_DEBUG("layer is %u, %u to %u link size %u", i, myRank, remoteRank, listSize);
            
            for (u32 idx = 0; idx < listSize; idx++) {
                if (protocol != linkList[idx].linkAttr.linkProtocol) {
                    continue;
                }

                HcclChannelDesc channelDesc;
                HcclChannelDescInit(&channelDesc, 1);
                channelDesc.remoteRank = remoteRank;
                CommLink link = linkList[idx];
                channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
                channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                HCCL_DEBUG("[CalcChannelRequestOmni] local device phyId: %u, remote device phyId: %u.",
                            channelDesc.localEndpoint.loc.device.devPhyId,
                            channelDesc.remoteEndpoint.loc.device.devPhyId);
                HCCL_INFO("[CalcChannelRequestOmni] Add channel request between %zu and %zu, netLayerIdx %u, "
                          "linkListIdx %u, protocol %zu",
                          myRank, channelDesc.remoteRank, i, idx, channelDesc.remoteEndpoint.protocol);
                channelDesc.channelProtocol = link.linkAttr.linkProtocol;
                channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
                channels.push_back(channelDesc);
            }
        }   

        HCCL_DEBUG("myRank is %u, layer is %u, channels size is %u", myRank, i, channels.size());     
    }

    return HCCL_SUCCESS;
}

// 分别记录两个Die上的channel，构造rankGroup
HcclResult CcuTempOmni::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc>& channelDescs)
{
    for (const auto &channel : channelDescs) {
        const RankId remoteRank = channel.remoteRank;
        uint32_t dieId = 0;
        HcclResult ret = GetChannelDieId(comm, myRank_, channel, dieId);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CcuTempOmni][PartitionChannels] Rank[%d] channel to remoteRank[%d], Failed to "
                "get dieId. errNo[0x%016llx]", myRank_, remoteRank, HCCL_ERROR_CODE(ret)),
            ret);
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempOmni][PartitionChannels] Rank[%d] channel to remoteRank[%d], dieId[%u] is "
                "invalid.", myRank_, remoteRank, dieId),
            HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempOmni][PartitionChannels] Rank[%d] channel to remoteRank[%d], insert to "
            "channels at dieId[%u].", myRank_, remoteRank, dieId);
        channels_[dieId].emplace_back(channel);  // 记录此channel属于哪个die
        rankGroup_[dieId].push_back(remoteRank); // 记录此rank属于哪个die
    }

    uint32_t minChannels = std::min(channels_[0].size(), channels_[1].size());
    uint32_t maxChannels = std::max(channels_[0].size(), channels_[1].size());
    // 如果1die的话，这个判断不对
    // CHK_PRT_RET(minChannels + 1 != maxChannels,
    //     HCCL_ERROR("[CcuTempOmni][PartitionChannels] Rank[%d], Unexpected channels size, "
    //         "die0 channels[%u], die1 channels[%u].", myRank_, channels_[0].size(), channels_[1].size()),
    //     HcclResult::HCCL_E_PARA);
    HCCL_DEBUG("[CcuTempOmni][PartitionChannels] Rank[%d], die0 channels[%u], die1 channels[%u].", myRank_,
        channels_[0].size(), channels_[1].size());
    // keep myRank_ at last, sync with kernel
    if (channels_[0].size() == 0 && channels_[1].size() != 0) {
        rankGroup_[1].push_back(myRank_);
        return HcclResult::HCCL_SUCCESS;
    } else if (channels_[0].size() != 0 && channels_[1].size() == 0) {
        rankGroup_[0].push_back(myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    if (channels_[0].size() < channels_[1].size()) {
        rankGroup_[0].push_back(myRank_);
    } else {
        rankGroup_[1].push_back(myRank_);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempOmni::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest, const XmlInfo& xmlInfo)
{
    HCCL_DEBUG("[CalcRes] rankid [%u] begin", mySubCommRank_);

    resourceRequest.notifyNumOnMainThread = xmlInfo.resInfo.notifyNumOnMainThread;
    resourceRequest.slaveThreadNum = xmlInfo.resInfo.slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(xmlInfo.resInfo.notifyNumPerThread, 1);
    // resourceRequest.ccuKernelNum.push_back(DIE_NUM);        // kernel数量

    // 计算channel信息
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestOmni(comm, param, topoInfo, subCommRanks_, xmlInfo.resInfo.mapchannelInfo, channelDescs));
    CHK_RET(PartitionChannels(comm, channelDescs));
    resourceRequest.channels.emplace_back(channelDescs);

    std::vector<std::vector<OmniSendRecvInfo>> tmpSendRecvInfo;
    tmpSendRecvInfo.resize(DIE_NUM);
    for (auto& sendRecvInfo : xmlInfo.vecSendRecvInfo) {
        if (std::find(rankGroup_[0].begin(), rankGroup_[0].end(), sendRecvInfo.dstSliceInfo[0].remoteRank) != rankGroup_[0].end()) {
            tmpSendRecvInfo[0].push_back(sendRecvInfo);
        } else if (std::find(rankGroup_[1].begin(), rankGroup_[1].end(), sendRecvInfo.dstSliceInfo[0].remoteRank) != rankGroup_[1].end()){
            tmpSendRecvInfo[1].push_back(sendRecvInfo);
        } else {
            HCCL_ERROR("[CalcRes] sendRecvInfo.remoteRank is %u not in rankGroup", sendRecvInfo.dstSliceInfo[0].remoteRank);
        }
    }

    bool handleSelfRank = true;
    if ((rankGroup_[0].size() > rankGroup_[1].size() && rankGroup_[1].size() != 0) || rankGroup_[0].size() == 0) {
        handleSelfRank = false;
    }

    HCCL_DEBUG("randgroup [0] size %u, [1] size %u", rankGroup_[0].size(), rankGroup_[1].size());

    uint32_t dieNum = 0;
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {    // 2Die算法，需要执行两次
        if (rankGroup_[dieId].size() == 0) continue;
        dieNum++;
        
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelOmni>(arg);
        };

        auto kernelArg = std::make_shared<CcuKernelArgOmni>(myRank_, param, subCommRanks_, rankGroup_[dieId], handleSelfRank, tmpSendRecvInfo[dieId]);
        kernelInfo.kernelArg = kernelArg;
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempOmni][CalcRes] dieId=%u, channels=%llu, ccuKernelInfos=%llu",
            dieId, channels_[dieId].size(), resourceRequest.ccuKernelInfos.size());
    }

    resourceRequest.ccuKernelNum.push_back(dieNum);        // kernel数量

    HCCL_DEBUG("[CcuTempOmni::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    HCCL_DEBUG("[CalcRes] rankid [%u] end", mySubCommRank_);

    return HcclResult::HCCL_SUCCESS;
}


HcclResult CcuTempOmni::KernelRun(const OpParam& param, 
                                            const TemplateDataParams& templateDataParams,
                                            const TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempOmni] rankid [%u] KernelRun begin", mySubCommRank_);


    buffInfo_ = templateDataParams.buffInfo;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;

    uint64_t sliceSize          = templateDataParams.sliceSize;
    uint64_t repeatNum          = templateDataParams.repeatNum;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;

    HCCL_INFO("templateResource.ccuKernels size %u", templateResource.ccuKernels.size());

    // 下发两个ccu kernel
    for(uint64_t dieIdx = 0; dieIdx < templateResource.ccuKernels.size(); dieIdx++) {
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgOmni>(
            inputAddr, outputAddr, scratchAddr, token, sliceSize, repeatNum, inputRepeatStride, outputRepeatStride);

        HCCL_INFO("[CcuTempOmni] [KernelRun] rankid [%u] input[%llu] outputAddr[%llu] scratchAddr[%llu] token[%llu] sliceSize[%llu] ,",
            mySubCommRank_, inputAddr, outputAddr, scratchAddr, token, sliceSize);

        void* taskArgPtr = static_cast<void*>(taskArg.get());
        HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[dieIdx], templateResource.ccuKernels[dieIdx], taskArgPtr);
    }

    HCCL_INFO("[CcuTempOmni] rankid [%u] KernelRun end", mySubCommRank_);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempOmni::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}
} // namespace ops_hccl