/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_NHR_1D_H
#define HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_NHR_1D_H

#include "ccu_alg_template_base.h"
#include "utils.h"
#include "ccu_kernel_all_gather_omnipipe_nhr_1d_mem2mem.h"

namespace ops_hccl {

class CcuTempAllGatherOmniPipeNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    explicit  CcuTempAllGatherOmniPipeNHR1DMem2Mem(const OpParam& param,
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempAllGatherOmniPipeNHR1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all gather ccu omnipipe mesh 1D with tempRankSize [%u].", subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;

    u64 GetThreadNum();
    HcclResult GetRes(AlgResourceRequest& resourceRequest);
    u64 CalcScratchSlice(u64 dataSize);
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    HcclResult GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum);
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult ProcessNHRStepInfo(HcclComm comm, const std::vector<HcclChannelDesc>& channelDescs,
                                  std::vector<NHRStepInfo>& stepInfoVector, std::map<u32, u32>& rank2ChannelIdx);
    HcclResult SplitDataFor2Dies(const OpParam& param, const TemplateDataParams& templateDataParams, uint64_t& die0Size,
                                 uint64_t& die1Size) const;

private:
    uint32_t mySubCommRank_ = 0;
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_NHR_1D_H