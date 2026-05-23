/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALLTOALL_PARALLEL_EXECUTOR_H
#define HCCLV2_INS_V2_ALLTOALL_PARALLEL_EXECUTOR_H

#include "executor_common_ops.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
class InsV2AlltoAllParallelExecutor : public InsCollAlgBase {
public:
    explicit InsV2AlltoAllParallelExecutor() = default;
    virtual ~InsV2AlltoAllParallelExecutor() = default;

    /* =============== 基类纯虚函数 =============== */

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm,
        TopoInfoWithNetLayerDetails* topoInfo,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcRes(
        HcclComm comm,
        const OpParam& param,
        const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        AlgResourceRequest& resourceRequest) override;

    HcclResult Orchestrate(
        const OpParam& param,
        const AlgResourceCtxSerializable& resCtx) override;

    HcclResult RestoreChannelMap(
        const AlgResourceCtxSerializable& resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const override;

private:
    /* =============== 参数生成 =============== */

    HcclResult GenAlgParamsStage1(
        const OpParam& param,
        u64 loopIdx,
        u64 loopSize,
        u64 scratchOffset,
        u64 inputDataOffset,
        TemplateDataParams& params);

    HcclResult GenAlgParamsStage2(
        const OpParam& param,
        u64 loopIdx,
        u64 loopSize,
        u64 scratchOffset,
        u64 outputDataOffset,
        TemplateDataParams& params);

    /* =============== 线程拆分 =============== */

    HcclResult PrepareResForTemplate(
        InsAlgTemplateX& templateX,
        InsAlgTemplateY& templateY,
        std::vector<ThreadHandle>& intraThreads,
        std::vector<ThreadHandle>& interThreads);

    /* =============== 编排循环 =============== */

    HcclResult OrchestrateLoop(
        const OpParam& param,
        InsAlgTemplateX& templateX,
        InsAlgTemplateY& templateY,
        const std::vector<std::map<u32, std::vector<ChannelInfo>>>& channelMapVec,
        std::vector<ThreadHandle>& intraThreads,
        std::vector<ThreadHandle>& interThreads);

    /* =============== 资源构建辅助 =============== */

    static TemplateResource BuildResource(
        const std::vector<ThreadHandle>& threads,
        const std::map<u32, std::vector<ChannelInfo>>& channels);

    /* =============== 成员变量 =============== */

    u64 xRankSize_{0};                              // X 维度组数
    u64 yRankSize_{0};                              // Y 维度组数
    u64 myXRank_{0};                                // 本 rank X 坐标
    u64 myYRank_{0};                                // 本 rank Y 坐标
    u64 scratchMultipleX_{0};                       // X 模板 scratch 倍数
    u64 scratchMultipleY_{0};                       // Y 模板 scratch 倍数
    AlgHierarchyInfoForAllLevel algHierarchyInfoCache_;  // 缓存的层级信息

    double multipleDimensionSplitRatio_{0.0};       // X/Y 数据拆分比例

    // 并行同步 (同 AllGather parallel executor 模式)
    ThreadHandle mainThread_;                        // 执行器主线程
    std::vector<ThreadHandle> templateMainThreads_;  // [0]=T0 main, [1]=T1 main
    std::vector<u32> syncNotifyOnTemplates_;         // main→template notify idx
    std::vector<u32> syncNotifyOnMain_;              // template→main notify idx

    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};

}  // namespace ops_hccl

#endif  // HCCLV2_INS_V2_ALLTOALL_PARALLEL_EXECUTOR_H
