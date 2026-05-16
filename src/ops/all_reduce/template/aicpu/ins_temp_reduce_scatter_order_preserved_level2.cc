/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_order_preserved_level2.h"

namespace ops_hccl {

InsTempReduceScatterOrderPreservedLevel2::InsTempReduceScatterOrderPreservedLevel2(const OpParam &param,
    const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    deterministicStrict_ = (param.deterministicConfig == DETERMINISTIC_STRICT);
    isUseCclIn_ = false;
}

InsTempReduceScatterOrderPreservedLevel2::~InsTempReduceScatterOrderPreservedLevel2()
{}

HcclResult InsTempReduceScatterOrderPreservedLevel2::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    threadNum = std::min(threadNum, static_cast<u32>(log2(templateRankSize_)) + 1);
    resourceRequest.slaveThreadNum = threadNum - 1;

    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(2);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel2][CalcRes] myRank[%u], threadNum[%u], "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, threadNum, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedLevel2::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 1;
}

HcclResult InsTempReduceScatterOrderPreservedLevel2::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    reduceOp_ = param.reduceType;

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel2][KernelRun] Start, threadNum[%u], count[%llu], "
        "dataType[%u], deterministicStrict[%d]", threadNum_, count_, dataType_, deterministicStrict_);

    CHK_RET(LocalCopy(tempAlgParams, templateResource.threads));

    CHK_RET(RunAllToAll(templateResource.channels, templateResource.threads, tempAlgParams));

    CHK_RET(RunLocalReduce(templateResource.threads, tempAlgParams));

    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel2][KernelRun] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel2::GetRes(AlgResourceRequest &resourceRequest) const
{
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedLevel2::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 0;
}

void InsTempReduceScatterOrderPreservedLevel2::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    for (u32 i = 0; i < threadNum_ - 1; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterOrderPreservedLevel2::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 i = 0; i < threadNum_ - 1; i++) {
        notifyIdxSubToMain.push_back(0);
    }
}

u32 InsTempReduceScatterOrderPreservedLevel2::CalcOutputIndex(const u32 round)
{
    return (round + myRank_) % templateRankSize_;
}

HcclResult InsTempReduceScatterOrderPreservedLevel2::LocalCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    }

    u64 sliceSize = tempAlgParams.allRankSliceSize.at(rankIdx);
    if (sliceSize == 0) {
        return HCCL_SUCCESS;
    }

    u64 srcOffset = tempAlgParams.sliceOffset.at(rankIdx);
    u32 outputIndex = CalcOutputIndex(rankIdx);
    u64 dstOffset = tempAlgParams.sliceOffset.at(outputIndex);

    HCCL_INFO("[LocalCopy] rankIdx[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        rankIdx, sliceSize, srcOffset, dstOffset);

    CHK_RET(static_cast<HcclResult>(HcommMemcpyAsync(threads[0],
        static_cast<u8 *>(buffInfo_.outputMem.addr) + dstOffset,
        static_cast<u8 *>(buffInfo_.outputMem.addr) + srcOffset,
        sliceSize, HCOMM_MEMCPY_KIND_D2D)));

    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel2::RunAllToAll(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunAllToAll] Start");

    for (u32 round = 0; round < templateRankSize_; round++) {
        if (round == myRank_) {
            continue;
        }

        ThreadHandle thread = threads[0];

        auto channelIter = channels.find(round);
        CHK_PRT_RET(channelIter == channels.end(),
            HCCL_ERROR("[RunAllToAll] channel not found for round[%u]", round), HCCL_E_INTERNAL);

        const std::vector<ChannelInfo> &channelInfos = channelIter->second;
        CHK_PRT_RET(channelInfos.empty(),
            HCCL_ERROR("[RunAllToAll] channelInfos empty for round[%u]", round), HCCL_E_INTERNAL);

        u64 sliceSize = tempAlgParams.allRankSliceSize.at(round);
        if (sliceSize == 0) {
            continue;
        }

        u64 srcOffset = tempAlgParams.sliceOffset.at(round);
        u32 outputIndex = CalcOutputIndex(round);
        u64 dstOffset = tempAlgParams.sliceOffset.at(outputIndex);

        for (const auto &channelInfo : channelInfos) {
            CHK_RET(static_cast<HcclResult>(HcommSendAsync(thread, channelInfo.linkHandle,
                static_cast<u8 *>(buffInfo_.outputMem.addr) + srcOffset, sliceSize)));

            CHK_RET(static_cast<HcclResult>(HcommRecvAsync(thread, channelInfo.linkHandle,
                static_cast<u8 *>(buffInfo_.outputMem.addr) + dstOffset, sliceSize)));
        }
    }

    HCCL_INFO("[RunAllToAll] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel2::RunLocalReduce(
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunLocalReduce] Start");

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

        CHK_RET(static_cast<HcclResult>(HcommReduceAsync(threads[0], srcMemPtr, dstMemPtr, count,
            dataType_, reduceOp_)));
    }

    HCCL_INFO("[RunLocalReduce] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel2::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempReduceScatterOrderPreservedLevel2", InsTempReduceScatterOrderPreservedLevel2);
}