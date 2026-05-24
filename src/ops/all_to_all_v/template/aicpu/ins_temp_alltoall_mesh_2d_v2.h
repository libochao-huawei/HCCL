/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALLTOALL_MESH_2D_V2_H
#define INS_TEMP_ALLTOALL_MESH_2D_V2_H

#include <atomic>

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

// Handles X-axis mesh ring exchange within a column of peers (yRankSize_ peers).
// Each rank in column rx communicates with all other column ranks (rx, ry') to
// exchange data destined for their respective destination rows.
//
// Inherits InsAlgTemplateBase directly (independent base class).
//
// v2.0 additions:
//   - RAII notify scope guard in KernelRun (Fix 3)
//   - Ring exchange timeout with peer failure bitmap (Fix 5)
//   - slaveErrs_[] array for per-slave error reporting (Fix 3)
class InsTempAlltoAllMesh2DV2 : public InsAlgTemplateBase {
public:
    InsTempAlltoAllMesh2DV2() = default;
    explicit InsTempAlltoAllMesh2DV2(const OpParam &param, const u32 rankId,
                                      const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllMesh2DV2() override;

    std::string Describe() const override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;

    virtual HcclResult GetRes(AlgResourceRequest &resourceRequest) const;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    virtual u64 GetThreadNum() const;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

protected:
    // Subclasses (MeshClosV2) override this for hash-based link selection
    virtual HcclResult RunAlltoAllMesh(
        const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels);

    // Local copy: own data from input → output + scratch for all destination slots
    virtual HcclResult LocalDataCopy(const std::vector<ThreadHandle> &threads);

    // Post copy: received data from scratch → output for all column peers
    HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads);

    u32 xRankSize_{0};     // total columns (from executor)
    u32 yRankSize_{0};     // total rows (from executor)
    u32 totalRankSize_{0}; // xRankSize_ * yRankSize_
    u32 myXRank_{0};       // rank's column index
    u32 myYRank_{0};       // rank's row index
    TemplateDataParams tempAlgParams_;

    // v2.0: fault tolerance
    std::vector<HcclResult> slaveErrs_;    // one per slave thread (Fix 3)
    // v3.0 Fix C: atomic peer bitmap for multi-link safety
    std::vector<std::atomic<bool>> failedRanks_;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALLTOALL_MESH_2D_V2_H
