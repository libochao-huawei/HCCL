/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_ORDER_PRESERVED_LEVEL2_H
#define INS_TEMP_ALL_GATHER_ORDER_PRESERVED_LEVEL2_H

#include <cstring>
#include <cmath>
#include <algorithm>
#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

class InsTempAllGatherOrderPreservedLevel2 : public InsAlgTemplateBase {
public:
    InsTempAllGatherOrderPreservedLevel2() = default;
    explicit InsTempAllGatherOrderPreservedLevel2(const OpParam &param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllGatherOrderPreservedLevel2() override;

    std::string Describe() const override
    {
        std::string info = "Template of Order Preserved AllGather Level2 (NHR/NB/Ring) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
        TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult RunAllGatherNHR(const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const TemplateDataParams &tempAlgParams);
    HcclResult RunAllGatherNB(const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const TemplateDataParams &tempAlgParams);
    HcclResult RunAllGatherRing(const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const TemplateDataParams &tempAlgParams);
    
    u32 GetStepNum(u32 rankSize) const;
    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, u32 &fromRank, u32 &toRank);
    
    TemplateDataParams tempAlgParams_;
    u64 count_{0};
    u32 dataTypeSize_{0};
    bool deterministicStrict_{false};
    std::string algType_;
};

}  // namespace ops_hccl

#endif