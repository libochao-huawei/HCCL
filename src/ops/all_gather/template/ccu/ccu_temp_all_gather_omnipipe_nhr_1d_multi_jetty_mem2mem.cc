/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_gather_omnipipe_nhr_1d_multi_jetty_mem2mem.h"
#include <map>
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"

namespace ops_hccl {

CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
    templateRankSize_ = ranks.size();
}

CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks,
    CommTopo priorityTopo)
    : CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem(param, rankId, subCommRanks)
{
    priorityTopo_ = priorityTopo;
}

CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::~CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem()
{
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    CHK_RET(GetRes(resourceRequest));
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs,
                                                  priorityTopo_));
    std::map<u32, u32> rank2ChannelIdx;
    for (u32 i = 0; i < channelDescs.size(); ++i) {
        u32 rankId = 0;
        CHK_RET(RemoteRankId2RankId(channelDescs[i].remoteRank, rankId));
        rank2ChannelIdx[rankId] = i;
    }
    std::vector<CcuOmniPipeNHRStepInfo> stepInfoVector;
    CHK_RET(CalcNHRInfo(stepInfoVector));
    resourceRequest.ccuKernelNum.push_back(stepInfoVector.size());
    for (const auto &stepInfo : stepInfoVector) {
        auto toChannel = rank2ChannelIdx.find(stepInfo.toRank);
        auto fromChannel = rank2ChannelIdx.find(stepInfo.fromRank);
        CHK_PRT_RET(toChannel == rank2ChannelIdx.end() || fromChannel == rank2ChannelIdx.end(),
                    HCCL_ERROR("[CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem][CalcRes] cannot find channel "
                               "for toRank[%u] or fromRank[%u].", stepInfo.toRank, stepInfo.fromRank),
                    HCCL_E_INTERNAL);
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem>(arg);
        };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem>(
            subCommRanks_[0].size(), myRank_, param, jettyNum_, stepInfo, toChannel->second, fromChannel->second,
            subCommRanks_);
        kernelInfo.channels = channelDescs;
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::CalcNHRInfo(
    std::vector<CcuOmniPipeNHRStepInfo> &stepInfoVector) const
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        CcuOmniPipeNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::GetNHRStepNum(u32 rankSize) const
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    return nSteps;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::GetStepInfo(
    u32 step, u32 nSteps, CcuOmniPipeNHRStepInfo &stepInfo) const
{
    u32 rankIdx = myRank_;
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx - deltaRank + templateRankSize_) % templateRankSize_;
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;
    stepInfo.nSlices = nSlices;
    stepInfo.toRank = (rankIdx + deltaRank) % templateRankSize_;
    stepInfo.fromRank = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::RemoteRankId2RankId(
    uint32_t remoteRankId, uint32_t &rankId) const
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    CHK_PRT_RET(it == ranks.end(),
                HCCL_ERROR("[CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem][RemoteRankId2RankId] "
                           "remoteRankId[%u] is not in subCommRanks.", remoteRankId),
                HCCL_E_PARA);
    rankId = std::distance(ranks.begin(), it);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams, TemplateResource &templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token = 0;
    CHK_RET(GetToken(buffInfo_, token));
    std::vector<CcuOmniPipeNHRStepInfo> stepInfoVector;
    CHK_RET(CalcNHRInfo(stepInfoVector));
    const StepSliceInfo &stepSliceInfo = templateDataParams.stepSliceInfo;
    for (u32 stepIdx = 0; stepIdx < stepInfoVector.size(); ++stepIdx) {
        CHK_PRT_RET(stepIdx >= templateResource.ccuKernels.size(),
                    HCCL_ERROR("[CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem][KernelRun] stepIdx[%u] exceeds "
                               "kernel size[%zu].", stepIdx, templateResource.ccuKernels.size()), HCCL_E_PARA);
        const auto &stepInfo = stepInfoVector[stepIdx];
        for (u32 idx = 0; idx < stepInfo.txSliceIdxs.size(); ++idx) {
            u32 sendSliceIdx = stepInfo.txSliceIdxs[idx];
            CHK_PRT_RET(sendSliceIdx >= stepSliceInfo.outputOmniPipeSliceStride.size() ||
                            sendSliceIdx >= stepSliceInfo.stepOutputSliceStride.size() ||
                            sendSliceIdx >= stepSliceInfo.stepSliceSize.size(),
                        HCCL_ERROR("[CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem][KernelRun] invalid "
                                   "sendSliceIdx[%u].", sendSliceIdx), HCCL_E_PARA);
            for (size_t rpt = 0; rpt < stepSliceInfo.stepSliceSize[sendSliceIdx].size(); ++rpt) {
                CHK_PRT_RET(rpt >= stepSliceInfo.outputOmniPipeSliceStride[sendSliceIdx].size(),
                            HCCL_ERROR("[CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem][KernelRun] invalid "
                                        "repeat[%zu] for sendSliceIdx[%u].", rpt, sendSliceIdx), HCCL_E_PARA);
                uint64_t srcOffset = stepSliceInfo.outputOmniPipeSliceStride[sendSliceIdx][rpt] +
                                     stepSliceInfo.stepOutputSliceStride[sendSliceIdx];
                uint64_t dstOffset = stepSliceInfo.outputOmniPipeSliceStride[sendSliceIdx][rpt] +
                                     stepSliceInfo.stepOutputSliceStride[sendSliceIdx];
                uint64_t sliceSize = stepSliceInfo.stepSliceSize[sendSliceIdx][rpt];
                uint64_t dataTypeSize = DataTypeSizeGet(param.DataDes.dataType);
                uint64_t dataCount = (dataTypeSize == 0) ? 0 : sliceSize / dataTypeSize;
                uint64_t sliceCountPerJetty = dataCount / jettyNum_ / (HCCL_MIN_SLICE_ALIGN / dataTypeSize) *
                                              (HCCL_MIN_SLICE_ALIGN / dataTypeSize);
                uint64_t sliceSizePerJetty = sliceCountPerJetty * dataTypeSize;
                uint64_t lastSliceSizePerJetty = sliceSize - sliceSizePerJetty * (jettyNum_ - 1);
                std::unique_ptr<hcomm::CcuTaskArg> taskArg =
                    std::make_unique<CcuTaskArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem>(
                        inputAddr, outputAddr, token, srcOffset, dstOffset, sliceSize, sliceSizePerJetty,
                        lastSliceSizePerJetty, idx == 0 ? 1 : 0,
                        idx == stepInfo.txSliceIdxs.size() - 1 ? 1 : 0);
                void *taskArgPtr = static_cast<void *>(taskArg.get());
                CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[stepIdx],
                                            taskArgPtr));
                CcuKernelSubmitInfo submitInfo;
                submitInfo.kernelHandle = templateResource.ccuKernels[stepIdx];
                CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token,
                                       srcOffset, dstOffset, sliceSize, sliceSizePerJetty, lastSliceSizePerJetty,
                                       idx == 0 ? 1 : 0, idx == stepInfo.txSliceIdxs.size() - 1 ? 1 : 0));
                templateResource.submitInfos.push_back(submitInfo);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::FastLaunch(
    const OpParam &param, const TemplateFastLaunchCtx &tempFastLaunchCtx)
{
    for (const auto &submitInfo : tempFastLaunchCtx.ccuKernelSubmitInfos) {
        const uint64_t *args = submitInfo.cachedArgs;
        CcuTaskArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem taskArg(
            PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[0],
            PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[1], args[2], args[3], args[4], args[5], args[6],
            args[7], args[8], args[9]);
        void *taskArgPtr = static_cast<void *>(&taskArg);
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0], submitInfo.kernelHandle,
                                    taskArgPtr));
    }
    return HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
