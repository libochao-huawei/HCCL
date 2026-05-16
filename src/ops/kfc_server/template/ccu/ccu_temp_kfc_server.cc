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
    HCCL_INFO("[CcuTempKfcServer::CalcRes]");
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
    (void)templateDataParams;
    (void)templateResource;
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