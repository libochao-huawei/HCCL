/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_clos_v3.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"
namespace ops_hccl {
namespace {
constexpr u32 COPY_THREAD_NUM = 1;
constexpr u32 COPY_NOTIFY_BASE_IDX = 1;
}

InsTempAlltoAllMeshClosV3::InsTempAlltoAllMeshClosV3(const OpParam &param, const u32 rankId,
                                                     const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAlltoAllMesh2DV3(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMeshClosV3::~InsTempAlltoAllMeshClosV3() {}

std::string InsTempAlltoAllMeshClosV3::Describe() const
{
    std::string info = "Template of alltoall MeshClosV2 (hash-based link selection) with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

u64 InsTempAlltoAllMeshClosV3::GetThreadNum() const
{
    return channelsPerRank_ + COPY_THREAD_NUM;
}

u32 InsTempAlltoAllMeshClosV3::GetCopyNotifySlotCount() const
{
    u32 commThreadNum = channelsPerRank_ == 0 ? 1 : channelsPerRank_;
    u32 peerNum = templateRankSize_ > 0 ? templateRankSize_ - 1 : 0;
    if (peerNum == 0) {
        return 1;
    }
    u32 stepNum = (peerNum + commThreadNum - 1) / commThreadNum;
    return stepNum * commThreadNum;
}

HcclResult InsTempAlltoAllMeshClosV3::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, COPY_NOTIFY_BASE_IDX);
        resourceRequest.notifyNumPerThread.back() = COPY_NOTIFY_BASE_IDX + GetCopyNotifySlotCount();
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV3][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAlltoAllMeshClosV3][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::RunAlltoAllMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] Entry: rank=%d templateRankSize=%u totalLinks=%u "
              "hierarchy: xRank=%u yRank=%u totalRank=%u myRank_=%u sliceSize=%llu",
              myRank_, templateRankSize_, channelsPerRank_,
              meshSize_, closSize_, rankSize_, myRank_,
              tempAlgParams_.sliceSize);
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(threads.size() < channelsPerRank_ + COPY_THREAD_NUM,
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] threads[%zu] < required[%u]. "
                           "commThreads=%u copyThreads=%u myRank=%d",
                           threads.size(), channelsPerRank_ + COPY_THREAD_NUM, channelsPerRank_, COPY_THREAD_NUM,
                           myRank_),
                HcclResult::HCCL_E_INTERNAL);

    std::vector<ThreadHandle> commThreads(threads.begin(), threads.begin() + channelsPerRank_);
    std::vector<ThreadHandle> copyThreads(threads.begin() + channelsPerRank_,
                                          threads.begin() + channelsPerRank_ + COPY_THREAD_NUM);
    u32 numSteps = (subCommRanks_[0].size() - 1 + commThreads.size() - 1) / commThreads.size();
    for (u32 step = 0; step < numSteps; step++) {
        for (u32 linkIdx = 0; linkIdx < commThreads.size(); linkIdx++) {
            CHK_RET(RunAlltoAllOnLink(commThreads, copyThreads, channels, linkIdx, step, numSteps));
        }
    }

    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] Failed rank[%u] detected. "
                       "templateRank=%u myRank=%d totalLinks=%u",
                       i, templateRankSize_, myRank_, channelsPerRank_);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::RunAlltoAllOnLink(
    const std::vector<ThreadHandle> &commThreads,
    const std::vector<ThreadHandle> &copyThreads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx, u32 step, u32 numSteps)
{
    CHK_PRT_RET(linkIdx >= commThreads.size(),
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] >= commThreads.size()[%zu] "
                           "myRank=%d templateRank=%u",
                           linkIdx, commThreads.size(), myRank_, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(copyThreads.empty(),
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] copyThreads is empty. "
                           "myRank=%d templateRank=%u",
                           myRank_, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    bool isPcie = IsPcieProtocol(channels);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 actualChunkSize = (totalSliceSize + rankSize_ - 1) / rankSize_;
    u64 chunkCount = actualChunkSize / dataTypeSize;
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Stride config: "
        "inputSliceStride=%llu outputSliceStride=%llu inBuffType=%d outBuffType=%d "
        "inBuffBaseOff=%llu outBuffBaseOff=%llu outputSize=%llu actualChunkSize=%llu",
        tempAlgParams_.inputSliceStride, tempAlgParams_.outputSliceStride,
        static_cast<int>(tempAlgParams_.buffInfo.inBuffType),
        static_cast<int>(tempAlgParams_.buffInfo.outBuffType),
        tempAlgParams_.buffInfo.inBuffBaseOff,
        tempAlgParams_.buffInfo.outBuffBaseOff,
        tempAlgParams_.buffInfo.outputSize,
        actualChunkSize);


    for (u32 peer = 0; peer < subCommRanks_[0].size(); ++peer) {
        u32 rankInSubcomm = subCommRanks_[0][peer];
        if (rankInSubcomm == myRank_) {
            continue;
        }
        if (channels.count(rankInSubcomm) == 0 || channels.at(rankInSubcomm).empty()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Peer %u has no channels. "
                       "myRank=%d linkIdx=%u channels.size=%zu templateRank=%u",
                       rankInSubcomm, myRank_, linkIdx, channels.size(), templateRankSize_);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];

        if ((myRank_ ^ connectedRank) % numSteps != step) {
            continue;
        }

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] rank[%d] peer[%u] already failed, skipping.",
                      linkIdx, myRank_, connectedRank);
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Rank[%d] connectedRank[%u] has no channels. "
                       "linkIdx=%u channels.size=%zu templateRank=%u",
                       myRank_, connectedRank, linkIdx, channels.size(), templateRankSize_);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 selectedLinkIdx = (myRank_ ^ connectedRank) % commThreads.size();

        if (selectedLinkIdx >= it->second.size()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] selectedLinkIdx OOB: "
                       "selectedLinkIdx=%u >= channels[%u].size()=%zu. myRank=%d connectedRank=%u",
                       selectedLinkIdx, connectedRank, it->second.size(), myRank_, connectedRank);
            continue;
        }

        if (!it->second[selectedLinkIdx].remoteCclMem.addr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] remoteCclMem.addr is NULL at selectedLinkIdx: "
                       "selectedLinkIdx=%u connectedRank=%u myRank=%d",
                       selectedLinkIdx, connectedRank, myRank_);
            continue;
        }

        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] matched: "
                  "myRank=%d connectedRank=%u selectedLinkIdx=%u/%u threads=%zu "
                  "enableRemoteMemAccess=%d isPcie=%d",
                  linkIdx, myRank_, connectedRank, selectedLinkIdx,
                  totalLinksToNeighbor, commThreads.size(),
                  enableRemoteMemAccess_, isPcie);

        if (linkIdx >= channels.at(connectedRank).size()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] channel index OOB: linkIdx[%u] >= "
                       "channels[%u].size()=%zu. myRank=%d templateRank=%u",
                       linkIdx, connectedRank, channels.at(connectedRank).size(), myRank_, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        if (!remoteCclBuffAddr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] remoteCclMem.addr is NULL for peer %u. "
                       "myRank=%d connectedRank=%u linkIdx=%u templateRank=%u",
                       connectedRank, myRank_, connectedRank, linkIdx, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        // tx 远端写
        void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
        u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff + connectedRank * actualChunkSize;
        txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, actualChunkSize, chunkCount);

        void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        u64 txDstOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * actualChunkSize;
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);

        // rx 远端读
        void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        u64 rxSrcOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * actualChunkSize;
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, actualChunkSize, chunkCount);
        
        void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
        u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * actualChunkSize;
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, actualChunkSize, chunkCount);

        HCCL_WARNING(
            "[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] rank[%d]->peer[%d] "
            "txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu "
            "actualSz=%llu",
            myRank_, connectedRank,
            txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset,
            actualChunkSize
        );

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        HcclResult dmaResult;
        if (isPcie) {
            // 远端读 因为是 1-1 配对的，需要在开始前 把对端要读的数据 加载到hccl buffer内。由于是对端的数据 所以是 connectedRank 
            u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff + connectedRank * actualChunkSize;
            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + connectedRank * actualChunkSize;
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, actualChunkSize, chunkCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, actualChunkSize, chunkCount);
            CHK_RET(LocalCopy(commThreads[linkIdx], srcSlice, dstSlice));

            dmaResult = SendRecvRead(sendRecvInfo, commThreads[linkIdx]);
        } else {
            dmaResult = SendRecvWrite(sendRecvInfo, commThreads[linkIdx]);

            // 远端写完成后只依赖当前 chunk，把后置 copy 放到专用线程，避免阻塞后续通信队列。
            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + connectedRank * actualChunkSize;
            u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * actualChunkSize;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, actualChunkSize, chunkCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, actualChunkSize, chunkCount);
            u32 notifyIdx = COPY_NOTIFY_BASE_IDX + step * commThreads.size() + linkIdx;
            CHK_RET(PreSyncInterThreads(commThreads[linkIdx], copyThreads, {notifyIdx}));
            CHK_RET(LocalCopy(copyThreads[0], srcSlice, dstSlice));
        }

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] peer %u timed out. "
                         "myRank=%d templateRank=%u",
                         linkIdx, connectedRank, myRank_, templateRankSize_);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos] send/recv FAILED: linkIdx=%u connectedRank=%u err=0x%x "
                       "myRank=%d templateRank=%u actualChunkSize=%llu",
                       linkIdx, connectedRank, dmaResult, myRank_, templateRankSize_, actualChunkSize);
            return dmaResult;
        }

    }
    return HCCL_SUCCESS;
}


HcclResult InsTempAlltoAllMeshClosV3::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    if (meshDataOffset > 0) {
        u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);
        DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    if (meshDataOffset < rankSize_ / meshSize_ - 1) {
        u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u32 reamindCount = (rankSize_ / meshSize_ - 1 - meshDataOffset) * meshDataOffset;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, cellSize * reamindCount, cellCount * reamindCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * reamindCount, cellCount * reamindCount);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * meshDataOffset;
    u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + cellSize * meshDataOffset;

    DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshSize_, cellCount * meshSize_);
    DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * meshSize_, cellCount * meshSize_);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    if (meshDataOffset > 0) {
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff;
        u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);
        
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    if (meshDataOffset < rankSize_ / meshSize_ - 1) {
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u32 reamindCount = (rankSize_ / meshSize_ - 1 - meshDataOffset) * meshDataOffset;

        DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * reamindCount, cellCount * reamindCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * reamindCount, cellCount * reamindCount);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
