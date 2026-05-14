/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_reduce_scatter_mesh_1D_write.h"
#include "log.h"
#include "channel.h"
#include "ccu_kernel_reduce_scatter_mesh1d_write.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"

namespace ops_hccl {

CcuTempReduceScatterMesh1DWrite::CcuTempReduceScatterMesh1DWrite(const OpParam &param, const u32 rankId,
                                                                    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempReduceScatterMesh1DWrite::~CcuTempReduceScatterMesh1DWrite()
{
}

HcclResult CcuTempReduceScatterMesh1DWrite::CalcRes(HcclComm comm, const OpParam &param,
                                                     const TopoInfoWithNetLayerDetails *topoInfo,
                                                     AlgResourceRequest &resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum        = 0;
    resourceRequest.ccuKernelNum.push_back(1);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelReduceScatterMesh1DWrite>(arg);
    };

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1DWrite>(
        subCommRanks_[0].size(), mySubCommRank_, param, subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceScatterMesh1DWrite::CalcRes] channelDescs.size()=%llu, dimsize=%llu",
               channelDescs.size(), subCommRanks_[0].size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1DWrite::KernelRun(const OpParam &param,
                                                       const TemplateDataParams &templateDataParams,
                                                       TemplateResource &templateResource)
{
    static uint32_t roundCounter = 0;
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t baseInputAddr = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddr    = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token         = hcomm::CcuRep::GetTokenInfo(baseInputAddr,
                                                          static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t sliceSize     = templateDataParams.sliceSize;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    // 穿刺版本：inputAddr 预先加上 mySubCommRank_ * inputSliceStride，传入已偏移的起始地址
    uint64_t inputAddr     = baseInputAddr + buffInfo_.inBuffBaseOff + mySubCommRank_ * inputSliceStride;

    HCCL_INFO("[ReduceScatterWrite] KernelRun round[%u], inputAddr[0x%llx], outputAddr[0x%llx], "
              "sliceSize[%llu], token[0x%llx], rank[%u/%lu], inputSliceStride[%llu]",
              roundCounter, inputAddr, outputAddr, sliceSize, token,
              mySubCommRank_, templateRankSize_, inputSliceStride);

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceScatterMesh1DWrite>(
        inputAddr, outputAddr, sliceSize, token);

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
                                  templateResource.ccuKernels[0], static_cast<void *>(taskArg.get())));

    HCCL_INFO("[ReduceScatterWrite] KernelRun round[%u] end", roundCounter);
    roundCounter++;
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterMesh1DWrite::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

} // namespace ops_hccl
