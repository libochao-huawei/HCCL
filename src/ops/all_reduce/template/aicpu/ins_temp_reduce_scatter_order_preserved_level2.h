/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_ORDER_PRESERVED_LEVEL2_H
#define INS_TEMP_REDUCE_SCATTER_ORDER_PRESERVED_LEVEL2_H

#include <cstring>
#include <cmath>
#include <algorithm>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempReduceScatterOrderPreservedLevel2 : public InsAlgTemplateBase {
public:
    InsTempReduceScatterOrderPreservedLevel2() = default;
    explicit InsTempReduceScatterOrderPreservedLevel2(const OpParam& param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterOrderPreservedLevel2() override;

    std::string Describe() const override
    {
        std::string info = "Template of Order Preserved ReduceScatter Level2 with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
        TemplateResource& templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult RunAllToAll(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams);
    HcclResult RunLocalReduce(const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult LocalCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    HcclResult PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    
    u32 CalcOutputIndex(const u32 round);
    
    u64 count_{0};
    u32 dataTypeSize_{0};
    bool isUseCclIn_{false};
    bool deterministicStrict_{false};
};

} // namespace ops_hccl

#endif