/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_order_preserved_level2.h"

namespace ops_hccl {

InsTempAllGatherOrderPreservedLevel2::InsTempAllGatherOrderPreservedLevel2(const OpParam &param,
    const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    deterministicStrict_ = (param.deterministicConfig == DETERMINISTIC_STRICT);
    algType_ = "NHR";
}

InsTempAllGatherOrderPreservedLevel2::~InsTempAllGatherOrderPreservedLevel2()
{}

HcclResult InsTempAllGatherOrderPreservedLevel2::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;

    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    HCCL_INFO("[InsTempAllGatherOrderPreservedLevel2][CalcRes] myRank[%u], threadNum[%u], "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u], algType[%s]",
        myRank_, threadNum, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, algType_.c_str());
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherOrderPreservedLevel2::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 1;
}

HcclResult InsTempAllGatherOrderPreservedLevel2::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    threadNum_ = templateResource.threads.size();
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    tempAlgParams_ = tempAlgParams;

    HCCL_INFO("[InsTempAllGatherOrderPreservedLevel2][KernelRun] Start, threadNum[%u], count[%llu], "
        "dataType[%u], deterministicStrict[%d], algType[%s]", 
        threadNum_, count_, dataType_, deterministicStrict_, algType_.c_str());

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    if (algType_ == "NHR") {
        CHK_RET(RunAllGatherNHR(templateResource.threads, templateResource.channels, tempAlgParams));
    } else if (algType_ == "NB") {
        CHK_RET(RunAllGatherNB(templateResource.threads, templateResource.channels, tempAlgParams));
    } else {
        CHK_RET(RunAllGatherRing(templateResource.threads, templateResource.channels, tempAlgParams));
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAllGatherOrderPreservedLevel2][KernelRun] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOrderPreservedLevel2::GetRes(AlgResourceRequest &resourceRequest) const
{
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherOrderPreservedLevel2::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 0;
}

void InsTempAllGatherOrderPreservedLevel2::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    for (u32 i = 0; i < threadNum_ - 1; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAllGatherOrderPreservedLevel2::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 i = 0; i < threadNum_ - 1; i++) {
        notifyIdxSubToMain.push_back(0);
    }
}

u32 InsTempAllGatherOrderPreservedLevel2::GetStepNum(u32 rankSize) const
{
    return rankSize - 1;
}

HcclResult InsTempAllGatherOrderPreservedLevel2::GetStepInfo(u32 step, u32 nSteps, u32 rank, 
    u32 rankSize, u32 &fromRank, u32 &toRank)
{
    u32 targetRank = (rank + step + 1) % rankSize;
    fromRank = targetRank;
    toRank = (rank + nSteps - step) % rankSize;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOrderPreservedLevel2::RunAllGatherNHR(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunAllGatherNHR] Start");

    if (templateRankSize_ == 1) {
        return HCCL_SUCCESS;
    }

    u32 nSteps = GetStepNum(templateRankSize_);

    for (u32 step = 0; step < nSteps; step++) {
        u32 fromRank = 0;
        u32 toRank = 0;
        CHK_RET(GetStepInfo(step, nSteps, myRank_, templateRankSize_, fromRank, toRank));

        auto fromChannelIter = channels.find(fromRank);
        CHK_PRT_RET(fromChannelIter == channels.end(),
            HCCL_ERROR("[RunAllGatherNHR] from channel not found"), HCCL_E_INTERNAL);

        auto toChannelIter = channels.find(toRank);
        CHK_PRT_RET(toChannelIter == channels.end(),
            HCCL_ERROR("[RunAllGatherNHR] to channel not found"), HCCL_E_INTERNAL);

        ThreadHandle thread = threads[0];

        for (const auto &channelInfo : fromChannelIter->second) {
            CHK_RET(static_cast<HcclResult>(HcommSendAck(thread, channelInfo.linkHandle)));
        }
        for (const auto &channelInfo : toChannelIter->second) {
            CHK_RET(static_cast<HcclResult>(HcommRecvAck(thread, channelInfo.linkHandle)));
        }

        u64 sliceSize = tempAlgParams.allRankSliceSize.at((myRank_ + step + 1) % templateRankSize_);
        u64 sliceOffset = tempAlgParams.sliceOffset.at((myRank_ + step + 1) % templateRankSize_);

        if (sliceSize > 0) {
            for (const auto &channelInfo : fromChannelIter->second) {
                CHK_RET(static_cast<HcclResult>(HcommRecvAsync(thread, channelInfo.linkHandle,
                    static_cast<u8 *>(buffInfo_.outputMem.addr) + sliceOffset, sliceSize)));
            }
            for (const auto &channelInfo : toChannelIter->second) {
                CHK_RET(static_cast<HcclResult>(HcommSendAsync(thread, channelInfo.linkHandle,
                    static_cast<u8 *>(buffInfo_.outputMem.addr) + sliceOffset, sliceSize)));
            }
        }

        for (const auto &channelInfo : fromChannelIter->second) {
            CHK_RET(static_cast<HcclResult>(HcommSendDataSignal(thread, channelInfo.linkHandle)));
        }
        for (const auto &channelInfo : toChannelIter->second) {
            CHK_RET(static_cast<HcclResult>(HcommRecvDataSignal(thread, channelInfo.linkHandle)));
        }
    }

    HCCL_INFO("[RunAllGatherNHR] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOrderPreservedLevel2::RunAllGatherNB(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunAllGatherNB] Start");
    return RunAllGatherNHR(threads, channels, tempAlgParams);
}

HcclResult InsTempAllGatherOrderPreservedLevel2::RunAllGatherRing(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunAllGatherRing] Start");
    return RunAllGatherNHR(threads, channels, tempAlgParams);
}

REGISTER_TEMPLATE_V2("InsTempAllGatherOrderPreservedLevel2", InsTempAllGatherOrderPreservedLevel2);
}