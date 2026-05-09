/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_gather_omnipipe_mesh_1d_mem2mem.h"
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "ccu_kernel_all_gather_omnipipe_mesh1d_mem2mem.h"

namespace ops_hccl {

CcuTempAllGatherOmniPipeMesh1DMem2Mem::CcuTempAllGatherOmniPipeMesh1DMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllGatherOmniPipeMesh1DMem2Mem::~CcuTempAllGatherOmniPipeMesh1DMem2Mem()
{
}

HcclResult CcuTempAllGatherOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam &param,
                                                          const TopoInfoWithNetLayerDetails *topoInfo,
                                                          AlgResourceRequest &resourceRequest)
{
    CHK_RET(GetRes(resourceRequest));
    resourceRequest.ccuKernelNum.push_back(1);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllGatherOmniPipeMesh1DMem2Mem>(arg);
    };
    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs,
                                                         CommTopo::COMM_TOPO_1DMESH));
        for (const auto &channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
                HCCL_ERROR("[CcuTempAllGatherOmniPipeMesh1DMem2Mem][CalcRes] channelProtocol[%u] invalid.",
                           channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem>(
        subCommRanks_[0].size(), myRank_, param, subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    HCCL_INFO("[CcuTempAllGatherOmniPipeMesh1DMem2Mem][CalcRes] ccu kernel resource ready.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeMesh1DMem2Mem::KernelRun(const OpParam &param,
                                                            const TemplateDataParams &templateDataParams,
                                                            TemplateResource &templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    const StepSliceInfo &stepSliceInfo = templateDataParams.stepSliceInfo;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token = 0;
    CHK_RET(GetToken(buffInfo_, token));

    CHK_PRT_RET(myRank_ >= stepSliceInfo.inputOmniPipeSliceStride.size() ||
                    myRank_ >= stepSliceInfo.outputOmniPipeSliceStride.size() ||
                    myRank_ >= stepSliceInfo.stepSliceSize.size(),
                HCCL_ERROR("[CcuTempAllGatherOmniPipeMesh1DMem2Mem] invalid myRank[%u] for stepSliceInfo.", myRank_),
                HCCL_E_PARA);

    const size_t rptNum = stepSliceInfo.stepSliceSize[myRank_].size();
    for (size_t rpt = 0; rpt < rptNum; ++rpt) {
        CHK_PRT_RET(rpt >= stepSliceInfo.inputOmniPipeSliceStride[myRank_].size() ||
                        rpt >= stepSliceInfo.outputOmniPipeSliceStride[myRank_].size(),
                    HCCL_ERROR("[CcuTempAllGatherOmniPipeMesh1DMem2Mem][KernelRun] invalid repeat[%zu] for "
                               "myRank[%u].", rpt, myRank_), HCCL_E_PARA);
        uint64_t srcOffset = stepSliceInfo.inputOmniPipeSliceStride[myRank_][rpt] +
                             stepSliceInfo.stepInputSliceStride[myRank_];
        uint64_t dstOffset = stepSliceInfo.outputOmniPipeSliceStride[myRank_][rpt] +
                             stepSliceInfo.stepOutputSliceStride[myRank_];
        uint64_t sliceSize = stepSliceInfo.stepSliceSize[myRank_][rpt];
        uint64_t isSrcDstEqual = (inputAddr + srcOffset == outputAddr + dstOffset) ? 1 : 0;
        std::unique_ptr<hcomm::CcuTaskArg> taskArg =
            std::make_unique<CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem>(inputAddr, outputAddr, token, srcOffset,
                                                                       dstOffset, sliceSize, isSrcDstEqual);
        void *taskArgPtr = static_cast<void *>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0],
                                    taskArgPtr));

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[0];
        CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token, srcOffset,
                               dstOffset, sliceSize, isSrcDstEqual));
        templateResource.submitInfos.push_back(submitInfo);
    }
    HCCL_INFO("[CcuTempAllGatherOmniPipeMesh1DMem2Mem][KernelRun] rptNum[%zu].", rptNum);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeMesh1DMem2Mem::FastLaunch(
    const OpParam &param, const TemplateFastLaunchCtx &tempFastLaunchCtx)
{
    for (const auto &submitInfo : tempFastLaunchCtx.ccuKernelSubmitInfos) {
        const uint64_t *args = submitInfo.cachedArgs;
        CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem taskArg(
            PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[0],
            PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[1], args[2], args[3], args[4], args[5],
            args[6]);
        void *taskArgPtr = static_cast<void *>(&taskArg);
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0], submitInfo.kernelHandle,
                                    taskArgPtr));
    }
    return HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherOmniPipeMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllGatherOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
