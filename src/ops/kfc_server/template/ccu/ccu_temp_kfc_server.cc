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
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "kernel/ccu_kernel_kfc_server.h"
#include "ccu_temp_kfc_server.h"

namespace ops_hccl {

CcuTempKfcServer::CcuTempKfcServer(const OpParam& param, const u32 rankId,
                                   const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

CcuTempKfcServer::~CcuTempKfcServer()
{
}

HcclResult CcuTempKfcServer::CalcChannelRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, std::vector<HcclChannelDesc>& channelDescs)
{
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    return HCCL_SUCCESS;
}

HcclResult CcuTempKfcServer::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                     AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempKfcServer::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;

    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelKfcServer>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRes(comm, param, topoInfo, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgKfcServer>(subCommRanks_[0].size(),
                                                                   mySubCommRank_,
                                                                   param.isMc2,
                                                                   param,
                                                                   subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempKfcServer::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempKfcServer::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempKfcServer::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[CcuTempKfcServer::FastLaunch] start");

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgKfcServer>(
        PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr),
        PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr));

    void* taskArgPtr = static_cast<void*>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0],
        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));
    HCCL_INFO("[CcuTempKfcServer::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempKfcServer::KernelRun(const OpParam& param,
                                       const TemplateDataParams& templateDataParams,
                                       TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempKfcServer] KernelRun");

    buffInfo_ = templateDataParams.buffInfo;

    TemplateAlgParams algParams;
    algParams.rankId = mySubCommRank_;
    algParams.rankSize = tempRankSize_;

    HCCL_INFO("[CcuTempKfcServer] KernelRun param inputPtr[%p], outputPtr[%p]",
              templateDataParams.buffInfo.inputPtr, templateDataParams.buffInfo.outputPtr);

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    submitInfo.kernelArg = templateResource.ccuKernels[0]->GetKernelArg();

    uint64_t inputAddr = PointerToAddr(templateDataParams.buffInfo.inputPtr);
    uint64_t outputAddr = PointerToAddr(templateDataParams.buffInfo.outputPtr);

    submitInfo.cachedArgs = new uint64_t[2];
    submitInfo.cachedArgs[0] = inputAddr;
    submitInfo.cachedArgs[1] = outputAddr;
    submitInfo.cachedArgsNum = 2;

    templateResource.submitInfos.push_back(submitInfo);

    HCCL_INFO("[CcuTempKfcServer] KernelRun End.");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempKfcServer::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

}