/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_reduce_mesh_1D_one_shot_write.h"
#include "log.h"
#include "channel.h"
#include "ccu_kernel_all_reduce_mesh1d_one_shot_write.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"

namespace ops_hccl {

CcuTempAllReduceMesh1DOneShotWrite::CcuTempAllReduceMesh1DOneShotWrite(const OpParam &param, const u32 rankId,
                                                                         const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
}

CcuTempAllReduceMesh1DOneShotWrite::~CcuTempAllReduceMesh1DOneShotWrite()
{
}

HcclResult CcuTempAllReduceMesh1DOneShotWrite::CalcRes(HcclComm comm, const OpParam &param,
                                                        const TopoInfoWithNetLayerDetails *topoInfo,
                                                        AlgResourceRequest &resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum        = 0;
    resourceRequest.ccuKernelNum.push_back(1);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllReduceMesh1DOneShotWrite>(arg);
    };

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1DOneShotWrite>(
        subCommRanks_[0].size(), mySubCommRank_, param, subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMesh1DOneShotWrite::CalcRes] channelDescs.size()=%llu, dimsize=%llu",
               channelDescs.size(), subCommRanks_[0].size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DOneShotWrite::KernelRun(const OpParam &param,
                                                          const TemplateDataParams &templateDataParams,
                                                          TemplateResource &templateResource)
{
    static uint32_t roundCounter = 0;
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr  = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token      = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t sliceSize  = templateDataParams.sliceSize;

    HCCL_INFO("[AllReduceWrite] KernelRun round[%u], inputAddr[0x%llx], outputAddr[0x%llx], "
              "sliceSize[%llu], token[0x%llx], rank[%u/%lu]",
              roundCounter, inputAddr, outputAddr, sliceSize, token, mySubCommRank_, templateRankSize_);

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceMesh1DOneShotWrite>(
        inputAddr, outputAddr, sliceSize, token);

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
                                  templateResource.ccuKernels[0], static_cast<void *>(taskArg.get())));

    HCCL_INFO("[AllReduceWrite] KernelRun round[%u] end", roundCounter);
    roundCounter++;
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceMesh1DOneShotWrite::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // write 模式 one-shot 场景，scratch Buffer 需要是 usrIn 的 rankSize 倍（与 read 模式一致）
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

} // namespace ops_hccl
