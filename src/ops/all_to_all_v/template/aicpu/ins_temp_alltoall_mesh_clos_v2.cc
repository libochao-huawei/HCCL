/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_clos_v2.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"

namespace ops_hccl {

InsTempAlltoAllMeshClosV2::InsTempAlltoAllMeshClosV2(const OpParam &param, const u32 rankId,
                                                     const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    // subCommRanks_[0] = Y-axis (Clos) group ranks
    yRankSize_ = subCommRanks_[0].size();

    for (u64 i = 0; i < yRankSize_; i++) {
        if (subCommRanks_[0][i] == rankId) {
            myYRingRank_ = i;
            break;
        }
    }

    HCCL_INFO("[InsTempAlltoAllMeshClosV2] Constructed. yRankSize_[%llu] myYRingRank_[%llu] rankId[%u].",
              yRankSize_, myYRingRank_, rankId);
}

InsTempAlltoAllMeshClosV2::~InsTempAlltoAllMeshClosV2() {}

u64 InsTempAlltoAllMeshClosV2::GetThreadNum() const
{
    return yChannelsPerRank_ > 0 ? (yChannelsPerRank_ + 1) : 1;
}

u64 InsTempAlltoAllMeshClosV2::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return yRankSize_;
}

HcclResult InsTempAlltoAllMeshClosV2::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV2][CalcRes] start.");
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, resourceRequest));
    CHK_RET(GetRes(resourceRequest));
    HCCL_INFO("[InsTempAlltoAllMeshClosV2][CalcRes] end. yChannelsPerRank_[%llu] yRankSize_[%llu].",
              yChannelsPerRank_, yRankSize_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = yChannelsPerRank_;
    resourceRequest.notifyNumOnMainThread = yChannelsPerRank_;
    resourceRequest.notifyNumPerThread.assign(yChannelsPerRank_, 1);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::CalcChannelRequestNhr(
    HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    yChannelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAlltoAllMeshClosV2][CalcChannelRequestNhr] totalLinks[%llu] channelCount[%zu].",
              yChannelsPerRank_, levelChannels.size());
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::KernelRun(const OpParam &param,
                                                 const TemplateDataParams &tempAlgParams,
                                                 TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV2][KernelRun] start. Rank[%u]", myRank_);

    CHK_RET(SetchannelsPerRank(templateResource.channels));
    yChannelsPerRank_ = channelsPerRank_;

    dataType_ = param.DataDes.dataType;
    myRank_ = param.userRank;
    isPcieProtocol_ = IsPcieProtocol(templateResource.channels);

    u64 perRankChunk = tempAlgParams.inputSliceStride;
    if (perRankChunk == 0) {
        HCCL_INFO("[InsTempAlltoAllMeshClosV2] perRankChunk is zero, skip.");
        return HCCL_SUCCESS;
    }
    allRankSize_ = tempAlgParams.sliceSize / perRankChunk;
    assert(allRankSize_ > 0 && tempAlgParams.sliceSize % perRankChunk == 0);
    xRankSize_ = allRankSize_ / yRankSize_;
    assert(xRankSize_ > 0 && allRankSize_ % yRankSize_ == 0);

    ThreadHandle mainThread = templateResource.threads[0];
    std::vector<ThreadHandle> subThreads(
        templateResource.threads.begin() + 1,
        templateResource.threads.end());

    CHK_RET(LocalDataCopy(mainThread, tempAlgParams));

    if (yRankSize_ > 1) {
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));
        CHK_RET(RunAlltoAllClosY(subThreads, templateResource.channels, tempAlgParams));
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }

    CHK_RET(PostLocalCopy(mainThread, tempAlgParams));

    HCCL_INFO("[InsTempAlltoAllMeshClosV2][KernelRun] end. Rank[%u]", myRank_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::RunAlltoAllClosY(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const TemplateDataParams &params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 perRankChunk = params.inputSliceStride;

    std::vector<u64> srcAddrs(allRankSize_), dstAddrs(allRankSize_);
    for (u32 i = 0; i < allRankSize_; i++) {
        srcAddrs[i] = params.buffInfo.inBuffBaseOff + perRankChunk * i;
        dstAddrs[i] = params.buffInfo.hcclBuffBaseOff + perRankChunk * i;
    }

    for (u32 linkIdx = 0; linkIdx < yChannelsPerRank_; linkIdx++) {
        if (linkIdx >= threads.size()) {
            HCCL_ERROR("[InsTempAlltoAllMeshClosV2][RunAlltoAllClosY] linkIdx[%u] >= threads.size()[%zu].",
                       linkIdx, threads.size());
            return HCCL_E_INTERNAL;
        }

        for (u64 neighborIdx = 0; neighborIdx < yRankSize_ - 1; neighborIdx++) {
            u64 remoteYRingRank = (myYRingRank_ + 1 + neighborIdx) % yRankSize_;
            u32 connectedAxisRank = static_cast<u32>(remoteYRingRank);
            u64 remoteRank = subCommRanks_[0][remoteYRingRank];

            auto it = channels.find(remoteRank);
            if (it == channels.end() || it->second.empty()) {
                continue;
            }

            u32 totalLinks = it->second.size();
            u32 selectedLinkIdx = (myYRingRank_ + remoteYRingRank) % totalLinks;

            if (selectedLinkIdx != linkIdx) {
                continue;
            }

            const ChannelInfo &linkRemote = it->second[selectedLinkIdx];
            std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;

            for (u32 i = 0; i < allRankSize_; i++) {
                bool toScratch = (params.buffInfo.outBuffType == BufferType::HCCL_BUFFER);

                if (i / xRankSize_ == connectedAxisRank) {
                    void *sPtr = toScratch ? params.buffInfo.inputPtr : params.buffInfo.hcclBuff.addr;
                    void *dPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                    u64 sc = perRankChunk / dataTypeSize;
                    u32 txDstGlobalIdx = connectedAxisRank * xRankSize_ + (i % xRankSize_);
                    txSrc.emplace_back(sPtr, srcAddrs[i], perRankChunk, sc);
                    txDst.emplace_back(dPtr, dstAddrs[txDstGlobalIdx], perRankChunk, sc);
                }
                if (i / xRankSize_ == myYRingRank_) {
                    void *rxSPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                    void *rxDPtr = toScratch ? params.buffInfo.hcclBuff.addr : params.buffInfo.outputPtr;
                    u64 sc = perRankChunk / dataTypeSize;
                    u32 rxSrcGlobalIdx = myYRingRank_ * xRankSize_ + (i % xRankSize_);
                    rxSrc.emplace_back(rxSPtr, srcAddrs[rxSrcGlobalIdx], perRankChunk, sc);
                    rxDst.emplace_back(rxDPtr, dstAddrs[i], perRankChunk, sc);
                }
            }

            if (txSrc.empty() && rxSrc.empty()) {
                continue;
            }

            TxRxSlicesList slicesList({txSrc, txDst}, {rxSrc, rxDst});
            TxRxChannels txRxChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(txRxChannels, slicesList);

            if (isPcieProtocol_) {
                CHK_RET(SendRecvRead(sendRecvInfo, threads[linkIdx]));
            } else {
                CHK_RET(SendRecvWrite(sendRecvInfo, threads[linkIdx]));
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::LocalDataCopy(
    const ThreadHandle &thread,
    const TemplateDataParams &params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 perRankChunk = params.inputSliceStride;
    u64 sliceCount = perRankChunk / dataTypeSize;

    u64 srcOffset = params.buffInfo.hcclBuffBaseOff + perRankChunk * myRank_;
    u64 dstOffset = params.buffInfo.outBuffBaseOff + params.outputSliceStride * myRank_;

    DataSlice srcSlice(params.buffInfo.hcclBuff.addr, srcOffset, perRankChunk, sliceCount);
    DataSlice dstSlice(params.buffInfo.outputPtr, dstOffset, perRankChunk, sliceCount);

    return LocalCopy(thread, srcSlice, dstSlice);
}

HcclResult InsTempAlltoAllMeshClosV2::PostLocalCopy(
    const ThreadHandle &thread,
    const TemplateDataParams &params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 perRankChunk = params.inputSliceStride;

    if (perRankChunk == 0) {
        return HCCL_SUCCESS;
    }

    u64 sliceCount = perRankChunk / dataTypeSize;
    u64 totalRanks = params.sliceSize / perRankChunk;

    for (u64 i = 0; i < totalRanks; i++) {
        u64 srcOffset = params.buffInfo.hcclBuffBaseOff + perRankChunk * i;
        u64 dstOffset = params.buffInfo.outBuffBaseOff + params.outputSliceStride * i;

        DataSlice srcSlice(params.buffInfo.hcclBuff.addr, srcOffset, perRankChunk, sliceCount);
        DataSlice dstSlice(params.buffInfo.outputPtr, dstOffset, perRankChunk, sliceCount);

        CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
    }

    return HCCL_SUCCESS;
}

void InsTempAlltoAllMeshClosV2::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.assign(yChannelsPerRank_, 0);
}

void InsTempAlltoAllMeshClosV2::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.resize(yChannelsPerRank_);
    for (u64 i = 0; i < yChannelsPerRank_; i++) {
        notifyIdxSubToMain[i] = i;
    }
}

HcclResult InsTempAlltoAllMeshClosV2::FastLaunch(const OpParam &param,
                                                  const TemplateFastLaunchCtx &tempFastLaunchCtx)
{
    (void)param;
    (void)tempFastLaunchCtx;
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl

REGISTER_TEMPLATE_V2("AlltoAllMeshClosV2", InsTempAlltoAllMeshClosV2)
