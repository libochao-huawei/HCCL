/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_RING_PIPELINE_H
#define INS_TEMP_ALL_GATHER_RING_PIPELINE_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

// ============================================================================
// AICPU Ring Pipeline AllGather 算法模板 v2
//
// 优化点（相比 v1）:
//   1. 通道过滤：只创建前驱+后继 2 条通道（原来用 Mesh1D 创建 N-1 条）
//   2. 选择器优化：考虑 inplace 场景、链路协议、rankSize 阈值
// ============================================================================

class InsTempAllGatherRingPipeline : public InsAlgTemplateBase {
public:
    InsTempAllGatherRingPipeline() = default;
    explicit InsTempAllGatherRingPipeline(const OpParam &param, const u32 rankId,
                                          const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllGatherRingPipeline() override;

    std::string Describe() const override
    {
        return StringFormat("Template of AllGather Ring Pipeline with tempRankSize [%u].",
                            templateRankSize_);
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

protected:
    HcclResult RunRingPipelineAllGather(const std::vector<ThreadHandle> &threads,
                                        const std::map<u32, std::vector<ChannelInfo>> &channels);
    HcclResult LocalDataCopy(const std::vector<ThreadHandle> &threads);
    HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads);
    u32 GetRankFromMap(const u32 algRankIdx) const;

    // Ring 拓扑
    u32 prevRank_ = 0;
    u32 nextRank_ = 0;
    u32 prevAlgRank_ = 0;
    u32 nextAlgRank_ = 0;

    TemplateDataParams tempAlgParams_;
    bool isDmaRead_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_GATHER_RING_PIPELINE_H
