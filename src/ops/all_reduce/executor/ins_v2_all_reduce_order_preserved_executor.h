/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_ORDER_PRESERVED_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_REDUCE_ORDER_PRESERVED_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

constexpr u64 HCCL_MIN_SLICE_ALIGN_A5 = 128;
constexpr u32 MIN_STRICT_RANK_NUM_A5 = 2;
constexpr u32 DEVICE_EIGHT_A5 = 8;
constexpr u32 DEVICE_FOUR_A5 = 4;

struct OrderPreservedMemInfo {
    u64 sizePerBlock;
    std::vector<u64> groupSize;
    bool scratchMemFlag;
    u64 totalSize;
    u32 all2allOffset;
};

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
class InsV2AllReduceOrderPreservedExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceOrderPreservedExecutor();
    ~InsV2AllReduceOrderPreservedExecutor() = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult InitExecutorInfo(const OpParam& param);
    HcclResult CalcSizePerBlock(const OpParam& param);
    HcclResult CalcGroupSlices(const OpParam& param);
    u64 RoundUpWithDivisor(u64 value, u64 divisor) const;
    u32 CalReduceStreamNum(const u32& localRankSize) const;
    
    HcclResult RunReduceScatterLevel1(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult RunReduceScatterLevel2(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult RunReduceScatterLevel1SingleRank(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult RunAllGatherLevel1(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult RunAllGatherLevel2(const OpParam &param, const AlgResourceCtxSerializable &resCtx);

    bool IsNeedStrictMode(const OpParam& param) const;
    bool CheckStrictCondition(const OpParam& param) const;

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

    OrderPreservedMemInfo memInfo_;
    bool deterministicStrict_{false};
    bool aicpuUnfoldMode_{false};
};
}

#endif