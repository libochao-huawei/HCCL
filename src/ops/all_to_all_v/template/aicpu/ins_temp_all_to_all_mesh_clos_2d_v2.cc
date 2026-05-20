/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_to_all_mesh_clos_2d_v2.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"

namespace ops_hccl {

InsTempAlltoAllMeshClos2DV2::InsTempAlltoAllMeshClos2DV2(const OpParam &param, const u32 rankId,
                                                         const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMeshClos2DV2::~InsTempAlltoAllMeshClos2DV2() {}

u64 InsTempAlltoAllMeshClos2DV2::GetThreadNum() const
{
    return xThreads_ + yThreads_;
}

u64 InsTempAlltoAllMeshClos2DV2::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

HcclResult InsTempAlltoAllMeshClos2DV2::CalcRes(HcclComm comm, const OpParam &param,
                                                 const TopoInfoWithNetLayerDetails *topoInfo,
                                                 AlgResourceRequest &resourceRequest)
{
    if (subCommRanks_.size() < COMM_LAYER_SIZE_2) {
        HCCL_WARNING("[InsTempAlltoAllMeshClos2DV2][CalcRes] subCommRanks size[%zu] < 2, fallback to single axis.",
                     subCommRanks_.size());
        xRankSize_ = subCommRanks_[0].size();
        yRankSize_ = 1;
    } else {
        xRankSize_ = subCommRanks_[0].size();
        yRankSize_ = subCommRanks_[1].size();
    }
    totalRankSize_ = xRankSize_ * yRankSize_;

    std::vector<HcclChannelDesc> xChannels;
    std::vector<std::vector<u32>> xSubComm = {subCommRanks_[0]};
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, xSubComm, xChannels));
    xChannelsPerRank_ = xChannels.empty() ? 1 : CalcChannelsPerRank(xChannels);

    std::vector<HcclChannelDesc> yChannels;
    if (yRankSize_ > 1) {
        std::vector<std::vector<u32>> ySubComm = {subCommRanks_[1]};
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, ySubComm, yChannels));
    }
    yChannelsPerRank_ = yChannels.empty() ? 1 : CalcChannelsPerRank(yChannels);

    std::vector<HcclChannelDesc> allChannels;
    allChannels.insert(allChannels.end(), xChannels.begin(), xChannels.end());
    allChannels.insert(allChannels.end(), yChannels.begin(), yChannels.end());
    resourceRequest.channels.push_back(allChannels);

    xThreads_ = (xRankSize_ > 1) ? (xRankSize_ - 1) * xChannelsPerRank_ : 1;
    yThreads_ = (yRankSize_ > 1) ? yChannelsPerRank_ : 0;

    u32 totalThreads = GetThreadNum();
    resourceRequest.slaveThreadNum = totalThreads > 1 ? totalThreads - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = totalThreads > 1 ? totalThreads - 1 : 0;

    HCCL_INFO("[InsTempAlltoAllMeshClos2DV2][CalcRes] xRs[%u] yRs[%u] totalRs[%u] "
              "xCh[%u] yCh[%u] xTh[%u] yTh[%u]",
              xRankSize_, yRankSize_, totalRankSize_,
              xChannelsPerRank_, yChannelsPerRank_, xThreads_, yThreads_);
    return HCCL_SUCCESS;
}

void InsTempAlltoAllMeshClos2DV2::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 t = GetThreadNum();
    for (u32 i = 1; i < t; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllMeshClos2DV2::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 t = GetThreadNum();
    for (u32 i = 0; i < t - 1; i++) {
        notifyIdxSubToMain.push_back(i);
    }
}

void InsTempAlltoAllMeshClos2DV2::ComputeStageAddrs(const TemplateDataParams &tp, u64 dataOffset, u64 dataSize,
                                                     bool fromInput, BufferType srcType, BufferType dstType,
                                                     std::vector<u64> &srcAddrs, std::vector<u64> &dstAddrs)
{
    (void)srcType;
    (void)dstType;
    srcAddrs.resize(totalRankSize_);
    dstAddrs.resize(totalRankSize_);

    for (u32 i = 0; i < totalRankSize_; i++) {
        u64 baseOff = fromInput ? tp.buffInfo.inBuffBaseOff : tp.buffInfo.hcclBuffBaseOff;
        srcAddrs[i] = baseOff + tp.sliceSize * i + dataOffset;

        u64 dstBase = fromInput ? tp.buffInfo.hcclBuffBaseOff : tp.buffInfo.outBuffBaseOff;
        dstAddrs[i] = dstBase + tp.sliceSize * i + dataOffset;
    }
}

HcclResult InsTempAlltoAllMeshClos2DV2::KernelRun(const OpParam &param,
                                                   const TemplateDataParams &tempAlgParams,
                                                   TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAlltoAllMeshClos2DV2][KernelRun] start. Rank[%d]", myRank_);

    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    sliceSize_ = tempAlgParams_.sliceSize;
    enableRemoteMemAccess_ = tempAlgParams_.enableRemoteMemAccess;

    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myXRank_));
    if (yRankSize_ > 1 && subCommRanks_.size() > 1) {
        CHK_RET(GetAlgRank(myRank_, subCommRanks_[1], myYRank_));
    } else {
        myYRank_ = 0;
    }

    u32 threadNum = GetThreadNum();
    if (templateResource.threads.size() < threadNum) {
        HCCL_ERROR("[InsTempAlltoAllMeshClos2DV2] threads[%zu] < needed[%u]",
                   templateResource.threads.size(), threadNum);
        return HCCL_E_INTERNAL;
    }

    u64 xSize = sliceSize_ / 2;
    u64 ySize = sliceSize_ - xSize;

    if (threadNum > 1) {
        std::vector<ThreadHandle> sub(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], sub, notifyIdxMainToSub_));
    }

    // ---- Stage 1: input → ccl buffer ----
    {
        std::vector<u64> srcAddrs, dstAddrs;
        ComputeStageAddrs(tempAlgParams_, 0, xSize, true,
                          BufferType::INPUT, BufferType::HCCL_BUFFER, srcAddrs, dstAddrs);
        CHK_RET(RunAxisMeshStage(templateResource.threads, templateResource.channels,
                                 subCommRanks_[0], xRankSize_, myXRank_, 0, xChannelsPerRank_,
                                 srcAddrs, dstAddrs, xSize, true,
                                 BufferType::INPUT, BufferType::HCCL_BUFFER));
    }
    {
        std::vector<u64> srcAddrs, dstAddrs;
        ComputeStageAddrs(tempAlgParams_, xSize, ySize, true,
                          BufferType::INPUT, BufferType::HCCL_BUFFER, srcAddrs, dstAddrs);
        if (yRankSize_ > 1) {
            CHK_RET(RunAxisMeshYClos(templateResource.threads, templateResource.channels,
                                     subCommRanks_[1], yRankSize_, myYRank_,
                                     xThreads_, yChannelsPerRank_,
                                     srcAddrs, dstAddrs, ySize,
                                     BufferType::INPUT, BufferType::HCCL_BUFFER));
        }
    }

    if (threadNum > 1) {
        std::vector<ThreadHandle> sub(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], sub, notifyIdxSubToMain_));
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], sub, notifyIdxMainToSub_));
    }

    // ---- Stage 2: ccl buffer → output ----
    {
        std::vector<u64> srcAddrs, dstAddrs;
        ComputeStageAddrs(tempAlgParams_, 0, xSize, false,
                          BufferType::HCCL_BUFFER, BufferType::OUTPUT, srcAddrs, dstAddrs);
        if (yRankSize_ > 1) {
            CHK_RET(RunAxisMeshYClos(templateResource.threads, templateResource.channels,
                                     subCommRanks_[1], yRankSize_, myYRank_,
                                     xThreads_, yChannelsPerRank_,
                                     srcAddrs, dstAddrs, xSize,
                                     BufferType::HCCL_BUFFER, BufferType::OUTPUT));
        }
    }
    {
        std::vector<u64> srcAddrs, dstAddrs;
        ComputeStageAddrs(tempAlgParams_, xSize, ySize, false,
                          BufferType::HCCL_BUFFER, BufferType::OUTPUT, srcAddrs, dstAddrs);
        CHK_RET(RunAxisMeshStage(templateResource.threads, templateResource.channels,
                                 subCommRanks_[0], xRankSize_, myXRank_, 0, xChannelsPerRank_,
                                 srcAddrs, dstAddrs, ySize, false,
                                 BufferType::HCCL_BUFFER, BufferType::OUTPUT));
    }

    if (threadNum > 1) {
        std::vector<ThreadHandle> sub(templateResource.threads.begin() + 1, templateResource.threads.end());
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], sub, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAlltoAllMeshClos2DV2][KernelRun] end. Rank[%d]", myRank_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClos2DV2::RunAxisMeshStage(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<u32> &axisRanks, u32 axisRankSize,
    u32 myAxisRank, u32 threadBaseIdx, u32 channelsPerRank,
    const std::vector<u64> &srcAddrs, const std::vector<u64> &dstAddrs,
    u64 dataSize, bool toScratch,
    BufferType srcType, BufferType dstType)
{
    if (dataSize == 0 || axisRankSize <= 1) {
        return HCCL_SUCCESS;
    }

    (void)srcType;
    (void)dstType;

    for (u32 neighborIdx = 0; neighborIdx < axisRankSize - 1; neighborIdx++) {
        u32 connectedRank = axisRanks[(myAxisRank + 1 + neighborIdx) % axisRankSize];
        u32 connectedAxisRank = (myAxisRank + 1 + neighborIdx) % axisRankSize;

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllMeshClos2DV2][RunAxisMeshStage] no channels for rank[%u]", connectedRank);
            return HCCL_E_INTERNAL;
        }

        u32 totalLinks = it->second.size();
        u32 selectedIdx = GetSelectedLinkIdx(connectedRank, totalLinks);
        const ChannelInfo &linkRemote = it->second[selectedIdx];

        u32 threadIdx = threadBaseIdx + neighborIdx * channelsPerRank + selectedIdx;
        if (threadIdx >= threads.size()) {
            HCCL_ERROR("[InsTempAlltoAllMeshClos2DV2] threadIdx[%u] >= threads[%zu]", threadIdx, threads.size());
            return HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;

        for (u32 i = 0; i < totalRankSize_; i++) {
            if (i % xRankSize_ == connectedAxisRank) {
                void *sPtr = toScratch ? tempAlgParams_.buffInfo.inputPtr : tempAlgParams_.buffInfo.hcclBuff.addr;
                void *dPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                u64 sc = dataSize / dataTypeSize_;
                u32 iy = i / xRankSize_;
                u32 txDstGlobalIdx = iy * xRankSize_ + myXRank_;
                txSrc.emplace_back(sPtr, srcAddrs[i], dataSize, sc);
                txDst.emplace_back(dPtr, dstAddrs[txDstGlobalIdx], dataSize, sc);
            }
            if (i % xRankSize_ == myXRank_) {
                void *sPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                void *dPtr = toScratch ? tempAlgParams_.buffInfo.hcclBuff.addr : tempAlgParams_.buffInfo.outputPtr;
                u64 sc = dataSize / dataTypeSize_;
                u32 iy = i / xRankSize_;
                u32 rxSrcGlobalIdx = iy * xRankSize_ + connectedAxisRank;
                rxSrc.emplace_back(sPtr, srcAddrs[rxSrcGlobalIdx], dataSize, sc);
                rxDst.emplace_back(dPtr, dstAddrs[i], dataSize, sc);
            }
        }

        if (txSrc.empty() && rxSrc.empty()) {
            continue;
        }

        TxRxSlicesList slices({txSrc, txDst}, {rxSrc, rxDst});
        TxRxChannels ch(linkRemote, linkRemote);
        SendRecvInfo info(ch, slices);
        CHK_PRT_RET(SendRecvRead(info, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAlltoAllMeshClos2DV2] SendRecvRead failed rank[%u]", connectedRank),
                    HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClos2DV2::RunAxisMeshYClos(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<u32> &axisRanks, u32 axisRankSize,
    u32 myAxisRank, u32 threadBaseIdx, u32 channelsPerRank,
    const std::vector<u64> &srcAddrs, const std::vector<u64> &dstAddrs,
    u64 dataSize, BufferType srcType, BufferType dstType)
{
    if (dataSize == 0 || axisRankSize <= 1) {
        return HCCL_SUCCESS;
    }
    (void)srcType;
    (void)dstType;

    for (u32 linkIdx = 0; linkIdx < channelsPerRank; linkIdx++) {
        u32 threadIdx = threadBaseIdx + linkIdx;
        if (threadIdx >= threads.size()) {
            HCCL_ERROR("[InsTempAlltoAllMeshClos2DV2][RunAxisMeshYClos] threadIdx[%u] >= threads[%zu]",
                       threadIdx, threads.size());
            return HCCL_E_INTERNAL;
        }

        for (u32 neighborIdx = 0; neighborIdx < axisRankSize - 1; neighborIdx++) {
            u32 connectedRank = axisRanks[(myAxisRank + 1 + neighborIdx) % axisRankSize];
            u32 connectedAxisRank = (myAxisRank + 1 + neighborIdx) % axisRankSize;

            auto it = channels.find(connectedRank);
            if (it == channels.end() || it->second.empty()) {
                continue;
            }

            u32 totalLinks = it->second.size();
            u32 selectedIdx = GetSelectedLinkIdx(connectedRank, totalLinks);
            if (selectedIdx != linkIdx) {
                continue;
            }

            const ChannelInfo &linkRemote = it->second[selectedIdx];

            std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;

            for (u32 i = 0; i < totalRankSize_; i++) {
                bool toScratch = (dstType == BufferType::HCCL_BUFFER);
                if (i / xRankSize_ == connectedAxisRank) {
                    void *sPtr = toScratch ? tempAlgParams_.buffInfo.inputPtr : tempAlgParams_.buffInfo.hcclBuff.addr;
                    void *dPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                    u64 sc = dataSize / dataTypeSize_;
                    u32 txDstGlobalIdx = connectedAxisRank * xRankSize_ + (i % xRankSize_);
                    txSrc.emplace_back(sPtr, srcAddrs[i], dataSize, sc);
                    txDst.emplace_back(dPtr, dstAddrs[txDstGlobalIdx], dataSize, sc);
                }
                if (i / xRankSize_ == myYRank_) {
                    void *sPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                    void *dPtr = toScratch ? tempAlgParams_.buffInfo.hcclBuff.addr : tempAlgParams_.buffInfo.outputPtr;
                    u64 sc = dataSize / dataTypeSize_;
                    u32 rxSrcGlobalIdx = myYRank_ * xRankSize_ + (i % xRankSize_);
                    rxSrc.emplace_back(sPtr, srcAddrs[rxSrcGlobalIdx], dataSize, sc);
                    rxDst.emplace_back(dPtr, dstAddrs[i], dataSize, sc);
                }
            }

            if (txSrc.empty() && rxSrc.empty()) {
                continue;
            }

            TxRxSlicesList slices({txSrc, txDst}, {rxSrc, rxDst});
            TxRxChannels ch(linkRemote, linkRemote);
            SendRecvInfo info(ch, slices);
            CHK_PRT_RET(SendRecvRead(info, threads[threadIdx]),
                        HCCL_ERROR("[InsTempAlltoAllMeshClos2DV2] YClos SendRecvRead failed rank[%u]", connectedRank),
                        HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
