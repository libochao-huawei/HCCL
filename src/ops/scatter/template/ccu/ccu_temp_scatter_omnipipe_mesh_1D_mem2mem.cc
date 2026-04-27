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
#include "ccu_kernel_scatter_omnipipe_mesh_1D_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_mesh_1D_mem2mem.h"

namespace ops_hccl {

CcuTempScatterOmniPipeMesh1DMem2Mem::CcuTempScatterOmniPipeMesh1DMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
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
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] templateRankSize[%u] subCommRootId_[%d]",
        __func__, rankId, mySubCommRank_, templateRankSize_, subCommRootId_);
}

CcuTempScatterOmniPipeMesh1DMem2Mem::~CcuTempScatterOmniPipeMesh1DMem2Mem()
{
}

void CcuTempScatterOmniPipeMesh1DMem2Mem::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempScatterOmniPipeMesh1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::vector<u32> ranks = subCommRanks_[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + " "; }
    HCCL_INFO("[CcuTempScatterOmniPipeMesh1DMem2Mem][SetRoot] ranks = subCommRanks[0] is: %s", ranksStr.c_str());
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
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
    return HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);

    HCCL_DEBUG("[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelScatterOmniPipeMesh1DMem2Mem>(arg);
    };

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, 
            myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto &channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[%s] Get Mesh Channel Success!", __func__);
    }

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgScatterOmniPipeMesh1DMem2Mem>(
        subCommRanks_[0].size(), mySubCommRank_, subCommRootId_, param, subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[%s]channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu", __func__,
        channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams, TemplateResource &templateResource)
{
    // if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
    //     HCCL_INFO("[%s] sliceSize is 0, no need to do, just success.", __func__);
    //     return HCCL_SUCCESS;
    // }

    buffInfo_ = templateDataParams.buffInfo;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = templateDataParams.stepSliceInfo.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = templateDataParams.stepSliceInfo.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t token = CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(buffInfo_.inputPtr), static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;

    uint64_t inputSliceStride = templateDataParams.inputSliceStride + stepSliceInfo.stepInputSliceStride[mySubCommRank_];
    uint64_t outputSliceStride = stepSliceInfo.stepOutputSliceStride[mySubCommRank_];
    uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();

    HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] repeatNum[%u] inputAddr[%llu] outputAddr[%llu]", 
        __func__, myRank_, mySubCommRank_, repeatNum, inputAddr, outputAddr);

    for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
        uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
        if (sliceSize == 0) {
            continue;
        }

        uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];
        uint64_t outputOmniPipeSliceStride = stepSliceInfo.outputOmniPipeSliceStride[mySubCommRank_][rpt];

        auto taskArg = std::make_unique<CcuTaskArgScatterOmniPipeMesh1DMem2Mem>(
            inputAddr, outputAddr, sliceSize, 0, token, 0,
            inputSliceStride, outputSliceStride, inputOmniPipeSliceStride,
            outputOmniPipeSliceStride);
        void *taskArgPtr = static_cast<void *>(taskArg.get());

        CHK_RET(HcclCcuKernelLaunch(
            param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

        HCCL_DEBUG("[%s] myRank[%u] rpt[%u] sliceSize[%llu] inputOmniPipeSliceStride[%llu] outputOmniPipeSliceStride[%llu]",
            __func__, myRank_, rpt, sliceSize, inputOmniPipeSliceStride, outputOmniPipeSliceStride);
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::FastLaunch(const OpParam &param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    // if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
    //     HCCL_INFO("[%s] ccu kernel num is 0, just success.", __func__);
    //     return HCCL_SUCCESS;
    // }

    // HCCL_DEBUG("[%s] start", __func__);
    // const uint64_t *args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    // buffInfo_ = tempFastLaunchCtx.buffInfo;

    // uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + args[0];
    // uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + args[1];
    // uint64_t sliceSize = args[2];
    // uint64_t token = args[3];
    // uint64_t inputSliceStride = args[4];
    // uint64_t outputSliceStride = args[5];
    // uint64_t inputOmniPipeSliceStride = args[6];
    // uint64_t outputOmniPipeSliceStride = args[7];
    // uint32_t rpt = static_cast<uint32_t>(args[8]);

    // CcuTaskArgScatterOmniPipeMesh1DMem2Mem taskArg(
    //     inputAddr, outputAddr, sliceSize, 0, token, 0,
    //     inputSliceStride, outputSliceStride, inputOmniPipeSliceStride,
    //     outputOmniPipeSliceStride, rpt);

    // void* taskArgPtr = static_cast<void*>(&taskArg);
    // CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0],
    //     tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));

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
