/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_V2_ALL_GATHER_V_SEQUENCE_EXECUTOR_H
#define INS_V2_ALL_GATHER_V_SEQUENCE_EXECUTOR_H

#include "executor_v2_base.h"
#include "alg_v2_template_base.h"

namespace ops_hccl {

/**
 * InsV2AllGatherVSequenceExecutor: Sequence executor for multi-layer AllGatherV.
 *
 * Runs InsAlgTemplate on each topology level sequentially:
 *   Level 0: AllGatherV within pod (intra-node mesh1D)
 *   Level 1: AllGatherV across pods (inter-node mesh1D, same local rank)
 *   Level 2: AllGatherV across supernodes (cross-supernode mesh1D, same sub-rank)
 *
 * Each level uses the same template type (InsAlgTemplate), following the
 * InsV2AllGatherSequenceExecutor pattern from all_gather.
 */
template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2AllGatherVSequenceExecutor : public InsCollAlgBase {
public:
    InsV2AllGatherVSequenceExecutor() {}
    ~InsV2AllGatherVSequenceExecutor() override {}

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                     AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

private:
    HcclResult InitCommInfo(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);

    uint32_t rankSizeLevel0_{0};
    uint32_t rankSizeLevel1_{0};
    uint32_t rankSizeLevel2_{0};
    uint32_t totalLevels_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};

}  // namespace ops_hccl

#endif