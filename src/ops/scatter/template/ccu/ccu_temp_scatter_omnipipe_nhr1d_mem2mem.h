/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H
#define HCCL_CCU_TEMP_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

class CcuTempScatterOmniPipeNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    CcuTempScatterOmniPipeNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>>& subCommRanks);
    ~CcuTempScatterOmniPipeNHR1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Scatter ccu OmniPipe NHR 1D Mem2Mem with tempRankSize[%u]", templateRankSize_);
    }

    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         TemplateResource& templateResource) override;

    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void SetRoot(u32 root);

private:
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo);
    HcclResult GetDieNumFromChannelDescs(HcclComm comm, u32& dieNum);
    HcclResult ProcessNHRStepInfo(HcclComm comm, std::vector<NHRStepInfo>& stepInfoVector,
                                  std::map<u32, u32>& rank2ChannelIdx, u32 enableDieNum,
                                  std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);
    HcclResult SplitDataFor2Dies(const OpParam& param, const uint64_t sliceSize,
                                 uint64_t& die0Size, uint64_t& die1Size) const;

    uint32_t mySubCommRank_ = 0;
    uint32_t subCommRootId_ = 0;
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
};

}  // namespace ops_hccl

#endif  // HCCL_CCU_TEMP_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H
