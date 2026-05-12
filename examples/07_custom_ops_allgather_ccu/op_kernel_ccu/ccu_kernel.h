/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "common.h"

namespace ops_hccl_ag {

struct CcuKernelArgAllGatherMesh1DMem2Mem : public CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
};

struct AllGatherMesh1DMem2MemContext {
    const CcuKernelArgAllGatherMesh1DMem2Mem* arg;
    
    CcuVariable input;
    std::vector<CcuVariable> output;
    std::vector<CcuVariable> token;
    CcuVariable currentRankSliceInputOffset;
    CcuVariable currentRankSliceOutputOffset;
    CcuVariable sliceSize;
    CcuEvent event;
};

CcuResult CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl_ag

#endif // HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H
