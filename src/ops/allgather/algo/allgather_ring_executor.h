/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ALLGATHER_RING_EXECUTOR_H
#define ALLGATHER_RING_EXECUTOR_H

#include "executor_base.h"
#include "coll_alg_exec_registry.h"

namespace ops_hccl {
// Ring 版 AllGather 执行器：
// 负责资源申请评估、执行编排、分片循环与模板算法调用。
class AllGatherRingExecutor : public ExecutorBase {
public:
    explicit AllGatherRingExecutor();
    ~AllGatherRingExecutor() override = default;

    // 计算算法资源需求（通道/线程/层级信息）。
    HcclResult CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
        AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType) override;
    // 执行编排入口：解析资源上下文并进入主循环。
    HcclResult Orchestrate(const OpParam &param, AlgResourceCtx* resCtx) override;

private:
    // 按可用 ccl buffer 大小分块循环执行。
    HcclResult RunLoop(const OpParam &param);
    // 单次分片执行：构建模板参数并下发 ring 内核流程。
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    // 生成等长切片描述，每个切片对应一个 rank 的目标区间。
    HcclResult PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice> &dataSlice) const;

    // 主线程句柄，用于在指定线程上下文提交通信原语。
    ThreadHandle thread_ = 0;
    // 分层通道信息：channels_[level][rank]。
    std::vector<std::vector<ChannelInfo>> channels_;
    // 当前使用的 level0 子通信域信息（ring 在 level0 运行）。
    SubCommInfo level0CommInfo_;
    // 单元素字节大小（由数据类型决定）。
    u32 unitSize_ = 0;
};
}

#endif
