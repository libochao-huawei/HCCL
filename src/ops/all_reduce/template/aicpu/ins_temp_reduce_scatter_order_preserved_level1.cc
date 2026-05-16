/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_order_preserved_level1.h"

namespace ops_hccl {

InsTempReduceScatterOrderPreservedLevel1::InsTempReduceScatterOrderPreservedLevel1(const OpParam &param,
    const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    deterministicStrict_ = (param.deterministicConfig == DETERMINISTIC_STRICT);
    all2allOffset_ = 0;
}

InsTempReduceScatterOrderPreservedLevel1::~InsTempReduceScatterOrderPreservedLevel1()
{}

HcclResult InsTempReduceScatterOrderPreservedLevel1::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    threadNum = std::min(threadNum, REDUCE_SCATTER_MAX_STREAM_NUM_A5);
    resourceRequest.slaveThreadNum = threadNum - 1;

    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(2);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][CalcRes] myRank[%u], threadNum[%u], "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, threadNum, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedLevel1::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    reduceOp_ = param.reduceType;

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][KernelRun] Start, threadNum[%u], count[%llu], "
        "dataType[%u], deterministicStrict[%d]", threadNum_, count_, dataType_, deterministicStrict_);

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(LocalCopy(tempAlgParams, templateResource.threads));

    CHK_RET(PreSyncThreads(templateResource.threads));

    CHK_RET(RunAllToAll(templateResource.channels, templateResource.threads, tempAlgParams));

    CHK_RET(PostSyncThreads(templateResource.threads));

    CHK_RET(RunLocalReduce(templateResource.threads, tempAlgParams));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][KernelRun] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::GetRes(AlgResourceRequest &resourceRequest) const
{
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedLevel1::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 0;
}

void InsTempReduceScatterOrderPreservedLevel1::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    for (u32 i = 0; i < threadNum_ - 1; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterOrderPreservedLevel1::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 i = 0; i < threadNum_ - 1; i++) {
        notifyIdxSubToMain.push_back(0);
    }
}

u32 InsTempReduceScatterOrderPreservedLevel1::CalcOutputIndex(const u32 round)
{
    return (all2allOffset_ + round + myRank_) % templateRankSize_;
}

bool InsTempReduceScatterOrderPreservedLevel1::IsLastBlockData(const u32 outputIndex)
{
    return outputIndex == templateRankSize_ - 1;
}

bool InsTempReduceScatterOrderPreservedLevel1::IsLastRank(const u32 rankId)
{
    return rankId == templateRankSize_ - 1;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::PreSyncThreads(const std::vector<ThreadHandle> &threads)
{
    if (threadNum_ > 1) {
        for (u32 i = 1; i < threadNum_; i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecord(threads[0], threads[i])));
        }
        for (u32 i = 1; i < threadNum_; i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWait(threads[i], threads[0])));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::PostSyncThreads(const std::vector<ThreadHandle> &threads)
{
    if (threadNum_ > 1) {
        for (u32 i = 1; i < threadNum_; i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecord(threads[i], threads[0])));
        }
        for (u32 i = 1; i < threadNum_; i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWait(threads[0], threads[i])));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::LocalCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[LocalCopy] myRank_ not found in subCommRanks_");
        return HCCL_E_INTERNAL;
    }

    u64 sliceSize = tempAlgParams.allRankSliceSize.at(rankIdx);
    if (sliceSize == 0) {
        return HCCL_SUCCESS;
    }

    u64 srcOffset = tempAlgParams.sliceOffset.at(rankIdx);

    void *dstMemPtr = nullptr;
    u32 outputIndex = CalcOutputIndex(rankIdx);
    u64 dstOffset = tempAlgParams.sliceOffset.at(outputIndex);

    HCCL_INFO("[LocalCopy] rankIdx[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu], outputIndex[%u]",
        rankIdx, sliceSize, srcOffset, dstOffset, outputIndex);

    CHK_RET(static_cast<HcclResult>(HcommMemcpyAsync(threads[0],
        static_cast<u8 *>(buffInfo_.outputMem.addr) + dstOffset,
        static_cast<u8 *>(buffInfo_.inputMem.addr) + srcOffset,
        sliceSize, HCOMM_MEMCPY_KIND_D2D)));

    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::RunAllToAll(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunAllToAll] Start");

    u32 streamIndex = 0;
    for (u32 round = 0; round < templateRankSize_; round++) {
        if (round == myRank_) {
            continue;
        }

        ThreadHandle thread = (streamIndex == 0) ? threads[0] : threads[streamIndex];

        auto channelIter = channels.find(round);
        CHK_PRT_RET(channelIter == channels.end(),
            HCCL_ERROR("[RunAllToAll] channel not found for round[%u]", round), HCCL_E_INTERNAL);

        const std::vector<ChannelInfo> &channelInfos = channelIter->second;
        CHK_PRT_RET(channelInfos.empty(),
            HCCL_ERROR("[RunAllToAll] channelInfos empty for round[%u]", round), HCCL_E_INTERNAL);

        u64 sliceSize = tempAlgParams.allRankSliceSize.at(round);
        if (sliceSize == 0) {
            streamIndex++;
            continue;
        }

        u64 srcOffset = tempAlgParams.sliceOffset.at(round);
        u32 outputIndex = CalcOutputIndex(round);
        u64 dstOffset = tempAlgParams.sliceOffset.at(outputIndex);

        for (const auto &channelInfo : channelInfos) {
            CHK_RET(static_cast<HcclResult>(HcommSendAsync(thread, channelInfo.linkHandle,
                static_cast<u8 *>(buffInfo_.inputMem.addr) + srcOffset, sliceSize)));

            CHK_RET(static_cast<HcclResult>(HcommRecvAsync(thread, channelInfo.linkHandle,
                static_cast<u8 *>(buffInfo_.outputMem.addr) + dstOffset, sliceSize)));
        }

        streamIndex++;
    }

    HCCL_INFO("[RunAllToAll] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::RunLocalReduce(
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunLocalReduce] Start, deterministicStrict[%d]", deterministicStrict_);

    u32 reduceStep = static_cast<u32>(std::ceil(log2(templateRankSize_)));

    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    }

    u64 srcOffset = tempAlgParams.sliceOffset.at(rankIdx);
    u64 sliceSize = tempAlgParams.allRankSliceSize.at(rankIdx);
    u64 count = sliceSize / dataTypeSize_;

    for (u32 step = 0; step < reduceStep; step++) {
        u32 stepSize = 1 << step;
        u32 peerRank = myRank_ ^ stepSize;

        if (peerRank >= templateRankSize_) {
            continue;
        }

        u64 peerSliceSize = tempAlgParams.allRankSliceSize.at(peerRank);
        if (peerSliceSize == 0) {
            continue;
        }

        u64 peerOffset = tempAlgParams.sliceOffset.at(peerRank);

        void *srcMemPtr = static_cast<u8 *>(buffInfo_.outputMem.addr) + peerOffset;
        void *dstMemPtr = static_cast<u8 *>(buffInfo_.outputMem.addr) + srcOffset;

        HCCL_INFO("[RunLocalReduce] step[%u], peerRank[%u], peerSliceSize[%llu], srcOffset[%llu], peerOffset[%llu]",
            step, peerRank, peerSliceSize, srcOffset, peerOffset);

        CHK_RET(static_cast<HcclResult>(HcommReduceAsync(threads[0], srcMemPtr, dstMemPtr, count,
            dataType_, reduceOp_)));
    }

    HCCL_INFO("[RunLocalReduce] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    }

    u64 sliceSize = tempAlgParams.allRankSliceSize.at(rankIdx);
    u64 srcOffset = tempAlgParams.sliceOffset.at(rankIdx);
    u64 dstOffset = 0;

    HCCL_INFO("[PostCopy] rankIdx[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        rankIdx, sliceSize, srcOffset, dstOffset);

    if (sliceSize > 0) {
        CHK_RET(static_cast<HcclResult>(HcommMemcpyAsync(threads[0],
            static_cast<u8 *>(buffInfo_.outputMem.addr) + dstOffset,
            static_cast<u8 *>(buffInfo_.outputMem.addr) + srcOffset,
            sliceSize, HCOMM_MEMCPY_KIND_D2D)));
    }

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempReduceScatterOrderPreservedLevel1", InsTempReduceScatterOrderPreservedLevel1);
}