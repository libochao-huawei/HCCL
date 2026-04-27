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
#include "ccu_kernel_reduce_scatter_omnipipe_mesh_1D.h"
#include "ccu_temp_reduce_scatter_omnipipe_mesh_1D.h"
#include "alg_data_trans_wrapper.h" // for localCopy in Template

namespace ops_hccl {

CcuTempReduceScatterOmniPipeMesh1D::CcuTempReduceScatterOmniPipeMesh1D(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] templateRankSize[%u]", __func__, rankId, mySubCommRank_, templateRankSize_);
}

CcuTempReduceScatterOmniPipeMesh1D::~CcuTempReduceScatterOmniPipeMesh1D()
{
}

u64 CcuTempReduceScatterOmniPipeMesh1D::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempReduceScatterOmniPipeMesh1D::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterOmniPipeMesh1D::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;

    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__, resourceRequest.notifyNumOnMainThread,
        resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelReduceScatterOmniPipeMesh1D>(arg);
    };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceScatterOmniPipeMesh1D>(
        subCommRanks_[0].size(), mySubCommRank_, param, subCommRanks_);
    // kernelInfo.channels = channelDescs;//[jjy][todo]
    kernelInfo.channels.push_back(channelDescs[0]);
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    resourceRequest.channels.push_back(channelDescs);

    HCCL_DEBUG("[%s]channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu", __func__, channelDescs.size(),
        subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    HCCL_DEBUG("[%s]ADDRCHK myRank[%u] mySubCommRank[%u] remoteRank[%u] loacalAddr[%u] remoteAddr[%u]", __func__,
        myRank_, mySubCommRank_, channelDescs[0].remoteRank, channelDescs[0].localEndpoint.commAddr.addr,
        channelDescs[0].remoteEndpoint.commAddr.addr);

    for (int i = 0; i < subCommRanks_.size(); i++) {
        for (int j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_DEBUG("hj aaa mySubCommRank_ is %u, myRank_ is %u, subCommRanks_[%d][%d] is %llu", mySubCommRank_,
                myRank_, i, j, subCommRanks_[i][j]);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterOmniPipeMesh1D::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams, TemplateResource &templateResource)
{
    HCCL_DEBUG("[%s] start", __func__);
    buffInfo_ = templateDataParams.buffInfo;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = templateDataParams.stepSliceInfo.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = templateDataParams.stepSliceInfo.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t token = CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(buffInfo_.inputPtr), static_cast<uint64_t>(buffInfo_.inputSize));
    HCCL_DEBUG("[%s] myRank[%u] pass", __func__, myRank_);
    HCCL_DEBUG("[%s] templateResource.threads size[%u] templateResource.ccuKernels size[%u]", __func__,
        templateResource.threads.size(), templateResource.ccuKernels.size());
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    uint64_t offset = 0;
    if (localCopyFlag == 0) {
        uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
        HCCL_DEBUG("[%s] myRank[%u] do ReduceScatter start", __func__, myRank_);
        uint64_t inputSliceStride = templateDataParams.inputSliceStride + stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint64_t outputSliceStride = stepSliceInfo.stepOutputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] repeatNum[%u]", __func__, myRank_, mySubCommRank_, repeatNum);

        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            if (sliceSize == 0) {
                continue;
            }
            uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];

            auto taskArg = std::make_unique<CcuTaskArgReduceScatterOmniPipeMesh1D>(inputAddr, outputAddr, sliceSize,
                offset, token, localCopyFlag, inputSliceStride, outputSliceStride, inputOmniPipeSliceStride);
            void *taskArgPtr = static_cast<void *>(taskArg.get());
            CHK_RET(HcclCcuKernelLaunch(
                param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
            HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] rpt[%u] inputAddrBase[%llu] outputAddrBase[%llu] "
                       "inBuffBaseOff[%llu] outBuffBaseOff[%llu] inputAddr[%llu] "
                       "outputAddr[%llu] sliceSize[%llu] sliceStride[%llu] localCopyFlag[%llu]",
                __func__, myRank_, mySubCommRank_, rpt, inputAddrBase, outputAddrBase, inBuffBaseOff, outBuffBaseOff,
                inputAddr, outputAddr, sliceSize, inputSliceStride, localCopyFlag);
        }
        HCCL_DEBUG("[%s] myRank[%u] do ReduceScatter end", __func__, myRank_);
    } else if (localCopyFlag == 1) {
#if T_DESC("Temp本地拷贝", true)
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
                   "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
#endif
// #if T_DESC("ccu本地拷贝", false)
//         uint64_t outputAddr = outputAddrBase;
//         HCCL_DEBUG("[%s] myRank[%u] CcuLocalCopy start", __func__, myRank_);
//         uint64_t sliceStride = templateDataParams.inputSliceStride;
//         uint64_t sliceSize = templateDataParams.sliceSize;
//         uint64_t inputOmniPipeSliceStride = 0;
//         auto taskArg = std::make_unique<CcuTaskArgReduceScatterOmniPipeMesh1D>(inputAddr, outputAddr, sliceSize,
//                 offset, token, localCopyFlag, sliceStride, 0, inputOmniPipeSliceStride);
//         void *taskArgPtr = static_cast<void *>(taskArg.get());
//         CHK_RET(HcclCcuKernelLaunch(
//             param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
//         HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] inputAddrBase[%llu] outputAddrBase[%llu] "
//                    "inBuffBaseOff[%llu] outBUffBaseOff[%llu] inputAddr[%llu] "
//                    "outputAddr[%llu] sliceSize[%llu] sliceStride[%llu] localCopyFlag[%llu]",
//             __func__, myRank_, mySubCommRank_, inputAddrBase, outputAddrBase, inBuffBaseOff, outBuffBaseOff, inputAddr,
//             outputAddr, sliceSize, sliceStride, localCopyFlag);
//         HCCL_DEBUG("[%s] myRank[%u] CcuLocalCopy start", __func__, myRank_);
// #endif
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterOmniPipeMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}
} // namespace ops_hccl