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
#include "ccu_kernel_gather_nhr1d_omnipipe_mem2mem.h"
#include "ccu_temp_gather_nhr_1D_omnipipe_mem2mem.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

CcuTempGatherNHR1DOmniPipeMem2Mem::CcuTempGatherNHR1DOmniPipeMem2Mem(const OpParam& param, const u32 rankId,
                                                                      const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }

    auto rootIt = std::find(ranks.begin(), ranks.end(), param.root);
    if (rootIt != ranks.end()) {
        mySubCommRoot_ = std::distance(ranks.begin(), rootIt);
    }

    HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem] mySubCommRank_=%u, mySubCommRoot_=%u, rankId=%u",
               mySubCommRank_, mySubCommRoot_, rankId);
}

CcuTempGatherNHR1DOmniPipeMem2Mem::~CcuTempGatherNHR1DOmniPipeMem2Mem()
{
}

void CcuTempGatherNHR1DOmniPipeMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        mySubCommRoot_ = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO("[CcuTempGatherNHR1DOmniPipeMem2Mem][SetRoot] mySubCommRank_=%u, root=%u, mySubCommRoot_=%u",
              mySubCommRank_, root, mySubCommRoot_);
}

u64 CcuTempGatherNHR1DOmniPipeMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

HcclResult CcuTempGatherNHR1DOmniPipeMem2Mem::ProcessNHRStepInfo(HcclComm comm,
                                                                  std::vector<GatherNHRStepInfo>& stepInfoVector,
                                                                  std::map<u32, u32>& rank2ChannelIdx,
                                                                  u32 enableDieNum, u32 enableDieId,
                                                                  std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    constexpr u32 DIE_NUM_1 = 1;
    constexpr u32 DIE_NUM_2 = 2;
    constexpr u32 DIE_0 = 0;
    constexpr u32 DIE_1 = 1;
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    
    for (u32 step = 0; step < nSteps; step++) {
        GatherNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        
        if (enableDieNum == DIE_NUM_1) {
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.fromRank, rankIdToChannelDesc_, enableDieId,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.toRank, rankIdToChannelDesc_, enableDieId,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
        } else if (enableDieNum == DIE_NUM_2) {
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.fromRank, rankIdToChannelDesc_, DIE_0,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.fromRank, rankIdToChannelDesc_, DIE_1,
                rank2ChannelIdx, channelsPerDie[DIE_1]));
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.toRank, rankIdToChannelDesc_, DIE_0,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.toRank, rankIdToChannelDesc_, DIE_1,
                rank2ChannelIdx, channelsPerDie[DIE_1]));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherNHR1DOmniPipeMem2Mem::FastLaunch(const OpParam& param,
                                                          const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempGatherNHR1DOmniPipeMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem::FastLaunch] start");
    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    const uint64_t* args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuTaskArgGatherNHROmniPipe1D taskArg(
            PointerToAddr(buffInfo_.inputPtr) + args[0],
            PointerToAddr(buffInfo_.outputPtr) + args[1],
            args[2], args[3], args[4]);

        void* taskArgPointer = static_cast<void*>(&taskArg);

        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0],
            tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPointer));
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }
    HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherNHR1DOmniPipeMem2Mem::CalcRes(HcclComm comm, const OpParam& param,
                                                       const TopoInfoWithNetLayerDetails* topoInfo,
                                                       AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    uint32_t enableDieNum = 0;
    uint32_t enableDieId = 0;
    CHK_RET(GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc_, myRank_, enableDieNum, enableDieId));

    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) {
        HCCL_ERROR("[CcuTempGatherNHR1DOmniPipeMem2Mem::CalcRes] get channelDescs fail, enableDieNum=%u", enableDieNum);
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u], kernelNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, kernelNum);

    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<GatherNHRStepInfo> stepInfoVector;

    CHK_RET(ProcessNHRStepInfo(comm, stepInfoVector, rank2ChannelIdx, enableDieNum, enableDieId, channelsPerDie));
    if (enableDieNum > 1) {
        CHK_RET(ReverseChannelPerDieIfNeed(comm, myRank_, channelsPerDie));
    }

    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg& arg) {
                                return std::make_unique<CcuKernelGatherNHROmniPipe1DMem2Mem>(arg);
                            };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgGatherNHROmniPipe1D>(
            subCommRanks_[0].size(), mySubCommRank_, mySubCommRoot_, kernelIdx,
            stepInfoVector, rank2ChannelIdx, param, subCommRanks_, enableDieNum);

        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherNHR1DOmniPipeMem2Mem::KernelRun(const OpParam& param,
                                                         const TemplateDataParams& templateDataParams,
                                                         TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempGatherNHR1DOmniPipeMem2Mem] Template KernelRun start.");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    u32 kernelNum = templateResource.ccuKernels.size();

    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    if (localCopyFlag == 1) {
        uint64_t sliceStride = templateDataParams.inputSliceStride;
        uint64_t sliceSize = templateDataParams.sliceSize;

        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgGatherNHROmniPipe1D>(
            inputAddr, outputAddr, token, sliceStride, localCopyFlag);

        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
            templateResource.ccuKernels[0], taskArgPtr));

        HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem::KernelRun] localCopy inputAddr=%llu outputAddr=%llu sliceSize=%llu",
                   inputAddr, outputAddr, sliceSize);

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[0];
        CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token,
                               sliceStride, localCopyFlag));
        templateResource.submitInfos.push_back(submitInfo);
    } else {
        uint64_t sliceStride = stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem::KernelRun] repeatNum=%u sliceStride=%llu", repeatNum, sliceStride);

        if (kernelNum > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            std::vector<u32> notifyIdxMainToSub(1, 0);
            CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
        }

        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            if (sliceSize == 0) {
                continue;
            }

            std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgGatherNHROmniPipe1D>(
                inputAddr, outputAddr, token, sliceStride, localCopyFlag);

            void* taskArgPtr = static_cast<void*>(taskArg.get());
            CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
                templateResource.ccuKernels[0], taskArgPtr));

            HCCL_DEBUG("[CcuTempGatherNHR1DOmniPipeMem2Mem::KernelRun] rpt=%u sliceSize=%llu sliceStride=%llu",
                       rpt, sliceSize, sliceStride);
        }

        if (kernelNum > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            std::vector<u32> notifyIdxSubToMain(1, 0);
            CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
        }

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[0];
        submitInfo.cachedArgs[0] = buffInfo_.inBuffBaseOff;
        submitInfo.cachedArgs[1] = buffInfo_.outBuffBaseOff;
        submitInfo.cachedArgs[2] = token;
        templateResource.submitInfos.push_back(submitInfo);
    }

    HCCL_INFO("[CcuTempGatherNHR1DOmniPipeMem2Mem] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherNHR1DOmniPipeMem2Mem::GetStepInfo(u32 step, u32 nSteps, GatherNHRStepInfo& stepInfo)
{
    u32 rankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = mySubCommRank_;

    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 sendTo = (rankIdx + deltaRank) % templateRankSize_;
    u32 recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;

    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = mySubCommRank_;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = ranks[sendTo];
    stepInfo.fromRank = ranks[recvFrom];

    HCCL_INFO("[CcuTempGatherNHR1DOmniPipeMem2Mem][GetStepInfo] nSlices[%u] toRank[%u] fromRank[%u]",
              nSlices, stepInfo.toRank, stepInfo.fromRank);

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_INFO("[CcuTempGatherNHR1DOmniPipeMem2Mem][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]",
                  i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempGatherNHR1DOmniPipeMem2Mem::GetThreadNum() const
{
    return 2;
}

HcclResult CcuTempGatherNHR1DOmniPipeMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl