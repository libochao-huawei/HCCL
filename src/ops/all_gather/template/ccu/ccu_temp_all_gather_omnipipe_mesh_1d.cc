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
#include "ccu_kernel_all_gather_omnipipe_mesh_1d.h"
#include "ccu_temp_all_gather_omnipipe_mesh_1d.h"

namespace ops_hccl {

CcuTempAllGatherOmniPipeMesh1D::CcuTempAllGatherOmniPipeMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks) // 这个里面会core
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }

    HCCL_DEBUG("mySubCommRank_ %u, rankId is %u", mySubCommRank_, rankId);
}

CcuTempAllGatherOmniPipeMesh1D::~CcuTempAllGatherOmniPipeMesh1D()
{
}

HcclResult CcuTempAllGatherOmniPipeMesh1D::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;

    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);

    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllGatherOmniPipeMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllGatherOmniPipeMesh1D>(arg);
                         };

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    for (int i = 0; i < subCommRanks_.size(); i++) {
        for (int j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_DEBUG("hj aaa mySubCommRank_ is %u, myRank_ is %u, subCommRanks_[%d][%d] is %llu", mySubCommRank_,
                myRank_, i, j, subCommRanks_[i][j]);
        }
    }

    for (HcclChannelDesc channel : channelDescs) {
        // uint32_t dieId = 0;
        // CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        // HCCL_DEBUG("aaa myRank_ is %u, dieId is %llu", myRank_, dieId);
        HCCL_DEBUG("hj channel mySubCommRank_ is %u, myRank_ is %u, remote rankid is %llu", mySubCommRank_, myRank_,
            channel.remoteRank);
    }

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherOmniPipeMesh1D>(
        subCommRanks_[0].size(), mySubCommRank_, param, subCommRanks_);
    // kernelInfo.channels = channelDescs;//[jjy][todo]
    kernelInfo.channels.push_back(channelDescs[0]);
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    resourceRequest.channels.push_back(channelDescs);

    HCCL_DEBUG("[CcuTempAllGatherOmniPipeMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeMesh1D::CalcScratchSlice(u64 dataSize)
{
    // mesh直接乘rankSize
    u64 scratchMultiple = templateRankSize_ * dataSize;
    return scratchMultiple;
}

u64 CcuTempAllGatherOmniPipeMesh1D::GetThreadNum()
{
    return 1;
}

HcclResult CcuTempAllGatherOmniPipeMesh1D::GetRes(AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeMesh1D::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    HCCL_DEBUG("[%s] start", __func__);
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    uint32_t rankId = myRank_;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = templateDataParams.stepSliceInfo.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = templateDataParams.stepSliceInfo.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token = CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(buffInfo_.inputPtr), static_cast<uint64_t>(buffInfo_.inputSize));

    if (localCopyFlag == 1) {
        uint64_t sliceStride = templateDataParams.inputSliceStride;
        uint64_t sliceSize = templateDataParams.sliceSize;
        uint64_t inputOmniPipeSliceStride = 0;
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGatherOmniPipeMesh1D>(inputAddr,
            outputAddr, token, sliceSize, sliceStride, localCopyFlag, inputOmniPipeSliceStride);
        void *taskArgPtr = static_cast<void *>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(
            param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
        HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] localCopy inputAddrBase[%llu] outputAddrBase[%llu] "
                   "inBuffBaseOff[%llu] "
                   "outBUffBaseOff[%llu] inputAddr[%llu] "
                   "outputAddr[%llu] sliceSize[%llu] localCopyFlag[%llu]",
            __func__, myRank_, mySubCommRank_, inputAddrBase, outputAddrBase, inBuffBaseOff, outBuffBaseOff, inputAddr,
            outputAddr, sliceSize, localCopyFlag);
    } else {
        uint64_t sliceStride = stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] repeatNum[%u]", __func__, myRank_, mySubCommRank_, repeatNum);
        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            if (sliceSize == 0) {
                continue;
            }

            uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];

            std::unique_ptr<hcomm::CcuTaskArg> taskArg
                = std::make_unique<CcuTaskArgAllGatherOmniPipeMesh1D>(inputAddr, outputAddr, token,
                    sliceSize, sliceStride, localCopyFlag, inputOmniPipeSliceStride);
            void *taskArgPtr = static_cast<void *>(taskArg.get());
            CHK_RET(HcclCcuKernelLaunch(
                param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
            HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] rpt[%u] inputAddrBase[%llu] outputAddrBase[%llu] "
                       "inBuffBaseOff[%llu] outBuffBaseOff[%llu] inputAddr[%llu] "
                       "outputAddr[%llu] sliceSize[%llu] sliceStride[%llu] localCopyFlag[%llu]",
                __func__, myRank_, mySubCommRank_, rpt, inputAddrBase, outputAddrBase, inBuffBaseOff, outBuffBaseOff,
                inputAddr, outputAddr, sliceSize, sliceStride, localCopyFlag);
        }
    }

    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}
} // namespace ops_hccl