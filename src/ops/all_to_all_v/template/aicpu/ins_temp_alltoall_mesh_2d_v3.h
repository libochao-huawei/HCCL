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

#include "ins_temp_alltoall_mesh_2d_v2.h"

namespace ops_hccl {

// V3 extension of Mesh2D template: supports variable port count per stage.
// Stage 1: 3-port mode (FM links only). Stage 2: 4-port mode (3 FM + 1 borrowed Clos).
//
// Key differences from V2:
//   - GetThreadNum() returns portCount_ (not templateRankSize_-1)
//   - GetRes() uses portCount_ for slave thread calculation
//   - SetPortCount() enables dynamic port count change between stages
//   - SetBorrowedLink() marks the borrowed link for bandwidth tracking
//   - LocalDataCopy/PostLocalCopy use outputSliceStride/inputSliceStride
//     for uniform cell spacing (cellSizeMax from executor)
class InsTempAlltoAllMesh2DV3 : public InsTempAlltoAllMesh2DV2 {
public:
    InsTempAlltoAllMesh2DV3() = default;
    explicit InsTempAlltoAllMesh2DV3(const OpParam &param, const u32 rankId,
                                      const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllMesh2DV3() override;

    // V3 overrides: thread count derived from port count, not template size.
    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;

    /// Set the number of active ports for this stage.
    /// Resets channelsPerRank_, slave thread count, and invalidates
    /// cached DMA offsets (recomputed on next KernelRun).
    /// Does NOT modify subCommRanks_, templateRankSize_, or scratch layout.
    void SetPortCount(u32 portCount);

    /// Mark a link as borrowed (e.g., Clos link borrowed by FM plane in Stage 2).
    /// @param isBorrowed  true if the given link index is borrowed
    /// @param linkIndex   which link index is borrowed (0-based)
    void SetBorrowedLink(bool isBorrowed, u32 linkIndex);

    /// Enable/disable shared port mode (not applicable to Mesh — no-op).
    /// Kept for interface uniformity with ClosV3.
    void SetSharedPortMode(bool enable);

protected:
    // V3: use executor-provided stride (cellSizeMax) for uniform cell spacing
    HcclResult LocalDataCopy(const std::vector<ThreadHandle> &threads) override;
    HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads) override;

    u32 portCount_ = 3;           // active port count (3 for Stage 1, 4 for Stage 2)
    bool hasBorrowedLink_ = false; // whether a borrowed link is active
    u32 borrowedLinkIndex_ = 0;   // index of the borrowed link
    bool sharedPortMode_ = false; // not used by mesh, but needed for interface
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALLTOALL_MESH_2D_V3_H
