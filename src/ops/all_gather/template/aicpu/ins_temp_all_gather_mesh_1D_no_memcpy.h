/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_MESH_1D_NO_MEMCPY_H
#define INS_TEMP_ALL_GATHER_MESH_1D_NO_MEMCPY_H

#include "ins_temp_all_gather_mesh_1D.h"

namespace ops_hccl {

class InsTempAllGatherMesh1DNoMemcpy : public InsTempAllGatherMesh1D {
public:
    InsTempAllGatherMesh1DNoMemcpy() = default;
    explicit InsTempAllGatherMesh1DNoMemcpy(const OpParam &param, const u32 rankId,
                                            const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllGatherMesh1DNoMemcpy() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather Mesh NoMemcpy with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

protected:
    HcclResult RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                const std::map<u32, std::vector<ChannelInfo>> &channels) override;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALL_GATHER_MESH_1D_NO_MEMCPY_H
