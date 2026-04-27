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
#include "ccu_kernel_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_nhr1d_mem2mem.h"

namespace ops_hccl {

CcuTempScatterOmniPipeNHR1DMem2Mem::CcuTempScatterOmniPipeNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                       const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();

    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }

    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }

    HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] templateRankSize[%u] subCommRootId_[%d]",
               __func__, rankId, mySubCommRank_, templateRankSize_, subCommRootId_);
}

CcuTempScatterOmniPipeNHR1DMem2Mem::~CcuTempScatterOmniPipeNHR1DMem2Mem()
{
}

void CcuTempScatterOmniPipeNHR1DMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }
    HCCL_DEBUG("[%s] Set root[%u] -> subCommRootId_[%d]", __func__, root, subCommRootId_);
}

u64 CcuTempScatterOmniPipeNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return subCommRanks_[0].size();
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::GetDieNumFromChannelDescs(HcclComm comm, u32& dieNum)
{
    constexpr u32 LINK_NUM_2 = 2;
    auto firstElement = rankIdToChannelDesc_.begin();
    if (firstElement == rankIdToChannelDesc_.end()) {
        HCCL_ERROR("[%s] rankIdToChannelDesc_ is empty", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const std::vector<HcclChannelDesc>& firstVector = firstElement->second;
    if (firstVector.size() == 1) {
        dieNum = 1;
        return HcclResult::HCCL_SUCCESS;
    } else if (firstVector.size() == LINK_NUM_2) {
        uint32_t dieId0 = 0;
        uint32_t dieId1 = 0;
        GetChannelDieId(comm, myRank_, firstVector[0], dieId0);
        GetChannelDieId(comm, myRank_, firstVector[1], dieId1);

        if (dieId0 == dieId1) {
            dieNum = 1;
        } else {
            dieNum = 2;
        }
        return HcclResult::HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[%s] get channelDescs fail: there are %zu links to rank %u",
                   __func__, firstVector.size(), firstElement->first);
        return HcclResult::HCCL_E_INTERNAL;
    }
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::ProcessNHRStepInfo(HcclComm comm,
                                                                  std::vector<NHRStepInfo>& stepInfoVector,
                                                                  std::map<u32, u32>& rank2ChannelIdx, u32 enableDieNum,
                                                                  std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);

        if (rank2ChannelIdx.count(stepInfo.fromRank) == 0 && stepInfo.rxSliceIdxs.size() != 0) {
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.fromRank] = curChannelIdx;
            for (HcclChannelDesc channel : rankIdToChannelDesc_.at(stepInfo.fromRank)) {
                uint32_t dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                uint32_t vecIdx = dieId % enableDieNum;
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }

        if (rank2ChannelIdx.count(stepInfo.toRank) == 0 && stepInfo.txSliceIdxs.size() != 0) {
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.toRank] = curChannelIdx;
            for (HcclChannelDesc channel : rankIdToChannelDesc_.at(stepInfo.toRank)) {
                u32 dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                u32 vecIdx = dieId % enableDieNum;
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param,
                                                       const TopoInfoWithNetLayerDetails* topoInfo,
                                                       AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    uint32_t enableDieNum = 0;
    CHK_RET(GetDieNumFromChannelDescs(comm, enableDieNum));

    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) {
        HCCL_ERROR("[%s] get channelDescs fail, enableDieNum=%u", __func__, enableDieNum);
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);

    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;

    CHK_RET(ProcessNHRStepInfo(comm, stepInfoVector, rank2ChannelIdx, enableDieNum, channelsPerDie));

    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg& arg) {
            return std::make_unique<CcuKernelScatterOmniPipeNHR1DMem2Mem>(arg);
        };
        auto kernelArg = std::make_shared<CcuKernelArgScatterOmniPipeNHR1DMem2Mem>(
            subCommRanks_[0].size(), mySubCommRank_, subCommRootId_, kernelIdx,
            enableDieNum, stepInfoVector, rank2ChannelIdx, param, subCommRanks_);
        kernelArg->channels = channelsPerDie[kernelIdx];
        kernelInfo.kernelArg = kernelArg;
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[%s] channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu",
               __func__, channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::SplitDataFor2Dies(const OpParam& param, const uint64_t sliceSize,
                                                                 uint64_t& die0Size, uint64_t& die1Size) const
{
    constexpr uint64_t MULTIPLIER = 4;
    uint64_t typeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t dataCount = sliceSize / typeSize;

    if (dataCount <= templateRankSize_ * MULTIPLIER) {
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HcclResult::HCCL_SUCCESS;
    }

    u8 die0PortGroupSize = 1;
    u8 die1PortGroupSize = 1;
    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * typeSize;
    die1Size = sliceSize - die0Size;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[%s] ccu kernel num is 0, just success.", __func__);
        return HCCL_SUCCESS;
    }

    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    const uint64_t* args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuTaskArgScatterOmniPipeNHR1DMem2Mem taskArg(
            PointerToAddr(buffInfo_.inputPtr) + args[0],
            PointerToAddr(buffInfo_.outputPtr) + args[1],
            args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
            args[10], args[11], args[12], args[13]);

        void* taskArgPtr = static_cast<void*>(&taskArg);
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[kernelIdx],
            tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].kernelHandle, taskArgPtr));
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::KernelRun(const OpParam& param,
                                                          const TemplateDataParams& templateDataParams,
                                                          TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[%s] sliceSize is 0, no need to do, just success.", __func__);
        return HCCL_SUCCESS;
    }

    buffInfo_ = templateDataParams.buffInfo;
    u32 kernelNum = templateResource.ccuKernels.size();

    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    uint64_t die0TailSize = 0;
    uint64_t die1TailSize = 0;
    constexpr uint32_t MAX_DIE_NUM_2 = 2;

    if (kernelNum == MAX_DIE_NUM_2) {
        SplitDataFor2Dies(param, templateDataParams.sliceSize, die0Size, die1Size);
        SplitDataFor2Dies(param, templateDataParams.tailSize, die0TailSize, die1TailSize);
    } else {
        die0Size = templateDataParams.sliceSize;
        die0TailSize = templateDataParams.tailSize;
    }

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;

    uint64_t inputOmniPipeSliceStride = 0;
    uint64_t outputOmniPipeSliceStride = 0;
    if (!templateDataParams.stepSliceInfo.stepInputSliceStride.empty()) {
        inputOmniPipeSliceStride = templateDataParams.stepSliceInfo.stepInputSliceStride[mySubCommRank_][0];
        outputOmniPipeSliceStride = templateDataParams.stepSliceInfo.stepOutputSliceStride[mySubCommRank_][0];
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {
        if ((axisId == 0 && die0Size == 0 && die0TailSize == 0) ||
            (axisId == 1 && die1Size == 0 && die1TailSize == 0)) {
            continue;
        }

        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgScatterOmniPipeNHR1DMem2Mem>(
            inputAddr, outputAddr, token, die0Size, die1Size, die0TailSize, die1TailSize,
            inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride,
            repeatNum, isInputOutputEqual, inputOmniPipeSliceStride, outputOmniPipeSliceStride);

        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId],
            templateResource.ccuKernels[axisId], taskArgPtr));
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    CcuKernelSubmitInfo submitInfo;
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token,
        die0Size, die1Size, die0TailSize, die1TailSize, inputSliceStride, outputSliceStride,
        inputRepeatStride, outputRepeatStride, repeatNum, isInputOutputEqual));

    for (u32 i = 0; i < kernelNum; i++) {
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = templateRankSize_;
    stepInfo.fromRank = templateRankSize_;
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    uint32_t rootId = subCommRootId_;
    u32 deltaRoot = (rootId + templateRankSize_ - virtRankIdx) % templateRankSize_;
    u32 deltaRankPair = 1 << step;

    u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    u32 nRanks = 0;
    bool isPowerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    if (!isPowerOfTwo && step == nSteps - 1) {
        nRanks = templateRankSize_ - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) {
        u32 sendTo = (virtRankIdx + templateRankSize_ - deltaRankPair) % templateRankSize_;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.txSliceIdxs.push_back(txSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.toRank = ranks[sendTo];
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) {
        u32 recvFrom = (virtRankIdx + deltaRankPair) % templateRankSize_;
        u32 rxSliceIdx = virtRankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.fromRank = ranks[recvFrom];
        stepInfo.nSlices = nSlices;
    }

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterOmniPipeNHR1DMem2Mem::GetThreadNum() const
{
    constexpr uint32_t KERNEL_NUM_2 = 2;
    return KERNEL_NUM_2;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
