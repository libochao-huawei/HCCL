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
#include "ccu_kernel_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_mesh1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

CcuTempScatterOmniPipeMesh1DMem2Mem::CcuTempScatterOmniPipeMesh1DMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    ifRealRoot_ = (rankId == param.root);
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + ", "; }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] subCommRanks[%s] templateRankSize[%u] subCommRootId_[%d] ifRealRoot_[%d]",
        __func__, rankId, mySubCommRank_, ranksStr.c_str(), templateRankSize_, subCommRootId_, ifRealRoot_);
}

CcuTempScatterOmniPipeMesh1DMem2Mem::~CcuTempScatterOmniPipeMesh1DMem2Mem()
{
}

void CcuTempScatterOmniPipeMesh1DMem2Mem::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempScatterOmniPipeMesh1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::string ranksStr = "";
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    for (auto r : ranks) { ranksStr += std::to_string(r) + ", "; }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] subCommRanks[%s] subCommRootId_[%d]",
        __func__, myRank_, mySubCommRank_,  ranksStr.c_str(), subCommRootId_);
}

void CcuTempScatterOmniPipeMesh1DMem2Mem::UnsetRoot(u32 rank)
{
    HCCL_INFO("[CcuTempScatterOmniPipeMesh1DMem2Mem][UnsetRoot] myRank_ [%u], unset root [%u] ", myRank_, rank);
    if (!ifRealRoot_) {
        subCommRootId_ = 1000;
    }
}

u64 CcuTempScatterOmniPipeMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

uint32_t CcuTempScatterOmniPipeMesh1DMem2Mem::RemoteRankId2RankId(const u32 remoteRankId) const
{
    u32 subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);

    HCCL_DEBUG("[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelScatterOmniPipeMesh1DMem2Mem>(arg);
    };

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
                HCCL_ERROR("[CcuTempScatterOmniPipeMesh1DMem2Mem][%s] channel.channelProtocol[%u]", __func__, channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }

    HCCL_DEBUG("[CcuTempScatterOmniPipeMesh1DMem2Mem][%s] Get Mesh channels Success.", __func__);
    std::map<u32, u32> subRankIdx2RankIdx;
    for (u32 i=0; i< channelDescs.size(); i++) {
        u32 remoteRank = channelDescs[i].remoteRank;
        u32 subRankIdx = RemoteRankId2RankId(remoteRank);
        subRankIdx2RankIdx[subRankIdx] = remoteRank;
    }
    subRankIdx2RankIdx[mySubCommRank_] = myRank_;

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgScatterOmniPipeMesh1DMem2Mem>(subCommRanks_[0].size(),
        mySubCommRank_, subCommRootId_, param, subCommRanks_, subRankIdx2RankIdx, ifRealRoot_, myRank_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    // resourceRequest.channels.push_back(channelDescs);

    HCCL_DEBUG("[%s]channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu", __func__, channelDescs.size(),
        subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    HCCL_DEBUG("[%s] myRank_[%u] mySubCommRank_[%u] remoteRank[%u] localAddr[%u] remoteAddr[%u]", __func__, myRank_,
        mySubCommRank_, channelDescs[0].remoteRank, channelDescs[0].localEndpoint.commAddr.addr,
        channelDescs[0].remoteEndpoint.commAddr.addr);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams, TemplateResource &templateResource)
{
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    HCCL_DEBUG("[%s] myRank[%u] mySubCommRank_[%u] isStepone[%d] isLastStep[%d] localCopyFlag[%d] start", __func__, myRank_,
        mySubCommRank_, isStepOne_, isLastStep_, localCopyFlag);
    buffInfo_ = templateDataParams.buffInfo;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);

    uint64_t outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    uint64_t inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t token = CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(buffInfo_.inputPtr), static_cast<uint64_t>(buffInfo_.inputSize));

    if (localCopyFlag == 0) {
        uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
        uint64_t inputSliceStride = 0;
        uint64_t outputSliceStride = 0;
        bool ifNewRoot = (subCommRootId_ == mySubCommRank_);
        uint64_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();

        HCCL_DEBUG("[%s] myRank[%u] mySubCommRank_[%u] ifNewRoot[%d] isStepone[%d] isLastStep[%d] repeatNum[%llu]", __func__, myRank_,
        mySubCommRank_, ifNewRoot, isStepOne_, isLastStep_, repeatNum);

        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = 0;
            uint64_t inputOmniPipeSliceStride = 0;
            uint64_t outputOmniPipeSliceStride = 0;
            if (ifRealRoot_) {
                sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
                inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];
                outputOmniPipeSliceStride= stepSliceInfo.outputOmniPipeSliceStride[mySubCommRank_][rpt];
            } else if (ifNewRoot) {
                sliceSize = stepSliceInfo.stepSliceSize[myRank_/templateRankSize_][rpt];
                inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[myRank_/templateRankSize_][rpt];
                outputOmniPipeSliceStride= stepSliceInfo.outputOmniPipeSliceStride[myRank_/templateRankSize_][rpt];
            }

            HCCL_INFO("[%s] myRank[%u] mySubCommRank[%u] subCommRootId_[%u] rpt[%u] sliceSize[%llu] "
                       "inputOmniPipeSliceStride[%llu] inputAddr[%llu], outputAddr[%llu] "
                       "outputOmniPipeSliceStride[%llu] isStepone[%d] isLastStep[%d]",
                __func__, myRank_, mySubCommRank_, subCommRootId_, rpt, sliceSize, inputOmniPipeSliceStride, inputAddr,
                outputAddr, outputOmniPipeSliceStride, isStepOne_, isLastStep_);
            
            auto taskArg = std::make_unique<CcuTaskArgScatterOmniPipeMesh1DMem2Mem>(
                inputAddr, outputAddr, sliceSize, 0, token, 0,
                inputSliceStride, outputSliceStride, inputOmniPipeSliceStride,
                outputOmniPipeSliceStride, isStepOne_, isLastStep_, ifNewRoot);
            void *taskArgPtr = static_cast<void *>(taskArg.get());

            CHK_RET(HcclCcuKernelLaunch(
                param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
        }
    } else if (localCopyFlag == 1) {
        HCCL_INFO("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_INFO("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
                   "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_INFO("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::FastLaunch(const OpParam &param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

} // namespace ops_hccl