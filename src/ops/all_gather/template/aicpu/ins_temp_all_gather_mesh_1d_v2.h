/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_MESH_1D_V2_H
#define INS_TEMP_ALL_GATHER_MESH_1D_V2_H

#include "ins_temp_all_gather_mesh_1D.h"

namespace ops_hccl {

class InsTempAllGatherMesh1DV2 : public InsTempAllGatherMesh1D {
public:
    InsTempAllGatherMesh1DV2() = default;
    explicit InsTempAllGatherMesh1DV2(const OpParam &param, const u32 rankId,
                                      const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllGatherMesh1DV2() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather Mesh1DV2 (4-port mode for Stage 2) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    u64 GetThreadNum() const override;

    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;

    void SetPortCount(u32 portCount) { portCount_ = portCount; }
    void SetBorrowEnabled(bool enabled) { borrowEnabled_ = enabled; }
    void SetDoubleLinkedNeighbor(u32 neighbor) { doubleLinkedNeighbor_ = neighbor; }

protected:
    HcclResult RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                const std::map<u32, std::vector<ChannelInfo>> &channels) override;

private:
    u32 portCount_ = 3;
    bool borrowEnabled_ = false;
    u32 doubleLinkedNeighbor_ = INVALID_VALUE_RANKID;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALL_GATHER_MESH_1D_V2_H
