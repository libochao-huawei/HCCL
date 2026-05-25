/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALLTOALL_MESH_CLOS_V2_H
#define INS_TEMP_ALLTOALL_MESH_CLOS_V2_H

#include "ins_temp_alltoall_mesh_2d_v2.h"

namespace ops_hccl {

class InsTempAlltoAllMeshClosV2 : public InsTempAlltoAllMesh2DV2 {
public:
    InsTempAlltoAllMeshClosV2() = default;
    explicit InsTempAlltoAllMeshClosV2(const OpParam &param, const u32 rankId,
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllMeshClosV2() override;

    std::string Describe() const override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;
    u64 GetThreadNum() const override;

protected:
    HcclResult RunAlltoAllMesh(
        const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels) override;

    HcclResult LocalDataCopy(const std::vector<ThreadHandle> &threads) override;
    HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads) override;

private:
    HcclResult RunAlltoAllOnLink(
        const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels,
        u32 linkIdx);
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALLTOALL_MESH_CLOS_V2_H
