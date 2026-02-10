/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_ALL_GATHER_SEQUENCE_EXECUTOR_H
#define INS_ALL_GATHER_SEQUENCE_EXECUTOR_H

#pragma once

#include "executor_v2_base.h"
#include "alg_v2_template_base.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllGatherSequenceExecutor : public InsCollAlgBase {
public:
    InsV2AllGatherSequenceExecutor() {}
    ~InsV2AllGatherSequenceExecutor() override {}
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfo* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

private:
    HcclResult InitCommInfo(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);

    uint32_t rankSizeLevel0_{0};
    uint32_t rankSizeLevel1_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};
}

#endif