/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_gather_omnipipe_local_copy.h"
#include "ccu_assist_pub.h"
#include "ccu_kernel_all_gather_omnipipe_local_copy.h"
#include "hccl_ccu_res.h"

namespace ops_hccl {

CcuTempAllGatherOmniPipeLocalCopy::CcuTempAllGatherOmniPipeLocalCopy(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    myRank_ = rankId;
    templateRankSize_ = subCommRanks.empty() ? 1 : subCommRanks[0].size();
}

HcclResult CcuTempAllGatherOmniPipeLocalCopy::CalcRes(HcclComm comm, const OpParam &param,
                                                      const TopoInfoWithNetLayerDetails *topoInfo,
                                                      AlgResourceRequest &resourceRequest)
{
    (void)comm;
    (void)topoInfo;
    CHK_RET(GetRes(resourceRequest));
    resourceRequest.ccuKernelNum.push_back(1);
    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllGatherOmniPipeLocalCopy>(arg);
    };
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherOmniPipeLocalCopy>(param, subCommRanks_);
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeLocalCopy::KernelRun(const OpParam &param,
                                                        const TemplateDataParams &templateDataParams,
                                                        TemplateResource &templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    CHK_PRT_RET(templateResource.threads.empty() || templateResource.ccuKernels.empty(),
                HCCL_ERROR("[CcuTempAllGatherOmniPipeLocalCopy][KernelRun] invalid template resource."),
                HCCL_E_PARA);
    uint64_t srcAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t dstAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t srcToken = 0;
    uint64_t dstToken = 0;
    BuffInfo srcBuffInfo;
    srcBuffInfo.inputPtr = buffInfo_.inputPtr;
    srcBuffInfo.inputSize = buffInfo_.inputSize;
    srcBuffInfo.hcclBuff = buffInfo_.hcclBuff;
    srcBuffInfo.hcclBuffSize = buffInfo_.hcclBuffSize;
    CHK_RET(GetToken(srcBuffInfo, srcToken));
    BuffInfo dstBuffInfo;
    dstBuffInfo.outputPtr = buffInfo_.outputPtr;
    dstBuffInfo.outputSize = buffInfo_.outputSize;
    dstBuffInfo.hcclBuff = buffInfo_.hcclBuff;
    dstBuffInfo.hcclBuffSize = buffInfo_.hcclBuffSize;
    CHK_RET(GetToken(dstBuffInfo, dstToken));
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGatherOmniPipeLocalCopy>(
        srcAddr, dstAddr, srcToken, dstToken, templateDataParams.inputSliceStride, templateDataParams.outputSliceStride,
        templateDataParams.sliceSize);
    void *taskArgPtr = static_cast<void *>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0],
                                taskArgPtr));

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, srcToken, dstToken,
                           templateDataParams.inputSliceStride, templateDataParams.outputSliceStride,
                           templateDataParams.sliceSize));
    templateResource.submitInfos.push_back(submitInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeLocalCopy::FastLaunch(const OpParam &param,
                                                         const TemplateFastLaunchCtx &tempFastLaunchCtx)
{
    for (const auto &submitInfo : tempFastLaunchCtx.ccuKernelSubmitInfos) {
        const uint64_t *args = submitInfo.cachedArgs;
        CcuTaskArgAllGatherOmniPipeLocalCopy taskArg(PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[0],
                                                     PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[1],
                                                     args[2], args[3], args[4], args[5], args[6]);
        void *taskArgPtr = static_cast<void *>(&taskArg);
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0], submitInfo.kernelHandle,
                                    taskArgPtr));
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeLocalCopy::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeLocalCopy::GetThreadNum() const
{
    return 1;
}

u64 CcuTempAllGatherOmniPipeLocalCopy::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

} // namespace ops_hccl
