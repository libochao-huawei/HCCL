/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALLTOALL_MESH_2D_V3_H
#define INS_TEMP_ALLTOALL_MESH_2D_V3_H

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
class InsTempAlltoAllMesh2DV3 : public InsAlgTemplateBase {
public:
    InsTempAlltoAllMesh2DV3() = default;
    explicit InsTempAlltoAllMesh2DV3(const OpParam &param, const u32 rankId,
                                      const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllMesh2DV3() override;

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
    // C-9 fix: virtual so clos can override with dy-based formula
    virtual HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads);

    u32 rankSize_{0};     
    u32 myRank_{0};   
    u32 meshSize_{0};    
    u32 closSize_{0};    
    TemplateDataParams tempAlgParams_;

    // peer failure bitmap (0=ok, 1=failed). uint8_t: non-copyable std::atomic
    // incompatible with std::vector assign/resize. Per-index single-writer
    // safety from disjoint hash subsets in ClosV2 multi-link ring.
    std::vector<uint8_t> failedRanks_;

public:
    // Public setter for 2D grid dimensions set by the parallel executor.
    // The template knows its 1D column peers from subCommRanks_ (x-axis),
    // but the full 2D grid (xRankSize, yRankSize, myXRank, myYRank) is only
    // known at the executor orchestration level.
    void SetMeshDimensions(u32 rankSize, u32 myRank, u32 meshSize, u32 closSize)
    {
        rankSize_ = rankSize;
        myRank_ = myRank;
        meshSize_ = meshSize;
        closSize_ = closSize;
    }

protected:
    // v2.0: fault tolerance
    std::vector<HcclResult> slaveErrs_;    // one per slave thread (Fix 3)
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALLTOALL_MESH_2D_V3_H
