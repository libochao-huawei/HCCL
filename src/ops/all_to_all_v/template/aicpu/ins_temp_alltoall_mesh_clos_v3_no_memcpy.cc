/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_clos_v3_no_memcpy.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"
#include <algorithm>
#include <vector>
namespace ops_hccl {
namespace {
constexpr u32 COPY_THREAD_NUM = 0;
constexpr u32 COPY_NOTIFY_BASE_IDX = 1;
constexpr u32 INVALID_GROUP_ID = static_cast<u32>(-1);

u32 GetPairwiseRoundNum(u32 groupNum)
{
    if (groupNum <= 1) {
        return 0;
    }
    return (groupNum % 2 == 0) ? groupNum - 1 : groupNum;
}

u32 GetPairGroupInRound(u32 groupNum, u32 myGroup, u32 round, bool &myGroupIsLeft)
{
    if (groupNum <= 1 || myGroup >= groupNum) {
        return INVALID_GROUP_ID;
    }

    const u32 scheduleGroupNum = (groupNum % 2 == 0) ? groupNum : groupNum + 1;
    const u32 roundNum = scheduleGroupNum - 1;
    const u32 dummyGroup = groupNum;
    std::vector<u32> groups(scheduleGroupNum);
    for (u32 idx = 0; idx < scheduleGroupNum; ++idx) {
        groups[idx] = idx;
    }

    for (u32 rotate = 0; rotate < round % roundNum; ++rotate) {
        u32 last = groups.back();
        for (u32 idx = scheduleGroupNum - 1; idx > 1; --idx) {
            groups[idx] = groups[idx - 1];
        }
        groups[1] = last;
    }

    for (u32 idx = 0; idx < scheduleGroupNum / 2; ++idx) {
        u32 left = groups[idx];
        u32 right = groups[scheduleGroupNum - 1 - idx];
        if (left == myGroup) {
            myGroupIsLeft = true;
            return right == dummyGroup ? INVALID_GROUP_ID : right;
        }
        if (right == myGroup) {
            myGroupIsLeft = false;
            return left == dummyGroup ? INVALID_GROUP_ID : left;
        }
    }
    return INVALID_GROUP_ID;
}
}

InsTempAlltoAllMeshClosV3NoMemcpy::InsTempAlltoAllMeshClosV3NoMemcpy(const OpParam &param, const u32 rankId,
                                                     const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAlltoAllMesh2DV3NoMemcpy(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMeshClosV3NoMemcpy::~InsTempAlltoAllMeshClosV3NoMemcpy() {}

std::string InsTempAlltoAllMeshClosV3NoMemcpy::Describe() const
{
    std::string info = "Template of alltoall MeshClosV2 (hash-based link selection) with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

u64 InsTempAlltoAllMeshClosV3NoMemcpy::GetThreadNum() const
{
    return channelsPerRank_ + COPY_THREAD_NUM;
}

u32 InsTempAlltoAllMeshClosV3NoMemcpy::GetCopyNotifySlotCount() const
{
    u32 commThreadNum = channelsPerRank_ == 0 ? 1 : channelsPerRank_;
    if (rankSize_ > 0 && meshSize_ > 0 && rankSize_ % meshSize_ == 0) {
        u32 colorRoundNum = GetPairwiseRoundNum(rankSize_ / meshSize_);
        u32 stepNum = (meshSize_ * colorRoundNum + commThreadNum - 1) / commThreadNum;
        return std::max(1u, stepNum * commThreadNum);
    }
    u32 peerNum = templateRankSize_ > 0 ? templateRankSize_ - 1 : 0;
    if (peerNum == 0) {
        return 1;
    }
    return std::max(1u, templateRankSize_ * commThreadNum);
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::GetRes(AlgResourceRequest &resourceRequest) const
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

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV3NoMemcpy][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAlltoAllMeshClosV3NoMemcpy][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::RunAlltoAllMesh(
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
    CHK_PRT_RET(meshSize_ == 0 || rankSize_ == 0 || rankSize_ % meshSize_ != 0,
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] invalid mesh dimensions. "
                           "myRank=%d rankSize=%u meshSize=%u closSize=%u",
                           myRank_, rankSize_, meshSize_, closSize_),
                HcclResult::HCCL_E_INTERNAL);
    u32 groupNum = rankSize_ / meshSize_;
    u32 colorRoundNum = GetPairwiseRoundNum(groupNum);
    u32 numSteps = (meshSize_ * colorRoundNum + static_cast<u32>(commThreads.size()) - 1) /
                   static_cast<u32>(commThreads.size());
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] schedule by mesh-group pairwise. "
                 "myRank=%d groupNum=%u meshSize=%u commThreads=%zu colorRounds=%u numSteps=%u",
                 myRank_, groupNum, meshSize_, commThreads.size(), colorRoundNum, numSteps);
    for (u32 step = 0; step < numSteps; step++) {
        for (u32 linkIdx = 0; linkIdx < commThreads.size(); linkIdx++) {
            CHK_RET(RunAlltoAllOnLink(commThreads, channels, linkIdx, step));
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

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::RunAlltoAllOnLink(
    const std::vector<ThreadHandle> &commThreads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx, u32 step)
{
    CHK_PRT_RET(linkIdx >= commThreads.size(),
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] >= commThreads.size()[%zu] "
                           "myRank=%d templateRank=%u",
                           linkIdx, commThreads.size(), myRank_, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);
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

    u32 groupNum = rankSize_ / meshSize_;
    u32 myGroup = myRank_ / meshSize_;
    u32 myLocalRank = myRank_ % meshSize_;
    u32 colorRoundNum = GetPairwiseRoundNum(groupNum);
    u32 microRound = step * static_cast<u32>(commThreads.size()) + linkIdx;
    u32 shift = colorRoundNum == 0 ? 0 : microRound / colorRoundNum;
    u32 colorRound = colorRoundNum == 0 ? 0 : microRound % colorRoundNum;
    if (shift >= meshSize_) {
        return HCCL_SUCCESS;
    }
    bool myGroupIsLeft = true;
    u32 peerGroup = GetPairGroupInRound(groupNum, myGroup, colorRound, myGroupIsLeft);
    if (peerGroup == INVALID_GROUP_ID) {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] dummy round. "
                     "myRank=%d step=%u linkIdx=%u micro=%u shift=%u color=%u groupNum=%u myGroup=%u",
                     myRank_, step, linkIdx, microRound, shift, colorRound, groupNum, myGroup);
        return HCCL_SUCCESS;
    }
    u32 connectedLocalRank = myGroupIsLeft ? (myLocalRank + shift) % meshSize_ :
                             (myLocalRank + meshSize_ - shift % meshSize_) % meshSize_;
    u32 connectedRank = peerGroup * meshSize_ + connectedLocalRank;

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] rank[%d] peer[%u] already failed, skipping.",
                      linkIdx, myRank_, connectedRank);
            return HCCL_SUCCESS;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Rank[%d] connectedRank[%u] has no channels. "
                       "linkIdx=%u channels.size=%zu templateRank=%u",
                       myRank_, connectedRank, linkIdx, channels.size(), templateRankSize_);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = static_cast<u32>(it->second.size());
        CHK_PRT_RET(totalLinksToNeighbor == 0,
                    HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] no effective link. "
                               "myRank=%d connectedRank=%u peerLinks=%u",
                               myRank_, connectedRank, totalLinksToNeighbor),
                    HcclResult::HCCL_E_INTERNAL);
        u32 selectedLinkIdx = linkIdx;

        if (selectedLinkIdx >= it->second.size()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] selectedLinkIdx OOB: "
                       "selectedLinkIdx=%u >= channels[%u].size()=%zu. myRank=%d connectedRank=%u "
                       "commThreads=%zu",
                       selectedLinkIdx, connectedRank, it->second.size(), myRank_, connectedRank,
                       commThreads.size());
            return HcclResult::HCCL_E_INTERNAL;
        }

        if (!it->second[selectedLinkIdx].remoteCclMem.addr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] remoteCclMem.addr is NULL at selectedLinkIdx: "
                       "selectedLinkIdx=%u connectedRank=%u myRank=%d peerLinks=%u",
                       selectedLinkIdx, connectedRank, myRank_, totalLinksToNeighbor);
            return HcclResult::HCCL_E_INTERNAL;
        }

        HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] matched: "
                  "myRank=%d connectedRank=%u step=%u micro=%u shift=%u color=%u myGroup=%u peerGroup=%u "
                  "myLocal=%u peerLocal=%u "
                  "selectedLinkIdx=%u/%u threads=%zu "
                  "enableRemoteMemAccess=%d isPcie=%d",
                  linkIdx, myRank_, connectedRank, step, microRound, shift, colorRound, myGroup, peerGroup,
                  myLocalRank, connectedLocalRank, selectedLinkIdx, totalLinksToNeighbor, commThreads.size(),
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

        CHK_PRT_RET(!enableRemoteMemAccess_ || linkRemote.remoteOutputGraphMode.addr == nullptr,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] remote output is unavailable. "
                               "myRank=%d connectedRank=%u linkIdx=%u enableRemoteMemAccess=%d",
                               myRank_, connectedRank, linkIdx, enableRemoteMemAccess_),
                    HcclResult::HCCL_E_INTERNAL);

        void *txDstPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 txDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
        CHK_PRT_RET(txSrcOffset + actualChunkSize > tempAlgParams_.buffInfo.inputSize ||
                        txDstOffset + actualChunkSize > linkRemote.remoteOutputGraphMode.size,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] slice out of registered range. "
                               "myRank=%d peer=%u txSrcOff=%llu txDstOff=%llu chunk=%llu "
                               "inputSize=%llu remoteOutputSize=%llu",
                               myRank_, connectedRank, txSrcOffset, txDstOffset, actualChunkSize,
                               tempAlgParams_.buffInfo.inputSize, linkRemote.remoteOutputGraphMode.size),
                    HcclResult::HCCL_E_INTERNAL);
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);

        // no-memcpy mode只使用远端写；rx slice仅用于SendRecvInfo占位。
        void *rxSrcPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 rxSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
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
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

        CHK_PRT_RET(isPcie,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] pcie/read protocol is not supported."),
                    HcclResult::HCCL_E_NOT_SUPPORT);
        HcclResult dmaResult = SendRecvWrite(sendRecvInfo, commThreads[linkIdx]);

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] peer %u timed out. "
                         "myRank=%d templateRank=%u",
                         linkIdx, connectedRank, myRank_, templateRankSize_);
            return HCCL_SUCCESS;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos] send/recv FAILED: linkIdx=%u connectedRank=%u err=0x%x "
                       "myRank=%d templateRank=%u actualChunkSize=%llu",
                       linkIdx, connectedRank, dmaResult, myRank_, templateRankSize_, actualChunkSize);
            return dmaResult;
        }

    return HCCL_SUCCESS;
}


HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::LocalDataCopy(const std::vector<ThreadHandle> &threads)
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

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::PostLocalCopy(const std::vector<ThreadHandle> &threads)
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
