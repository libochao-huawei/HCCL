/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H

#include <vector>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllToAllMesh2Die : CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    bool                                    withMyRank;
    uint32_t                                localSize;
    uint32_t                                localId;
    std::vector<uint32_t>                   rankGroup;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct AllToAllMesh2DieContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllMesh2Die *arg;

    uint64_t rankSize{0};
    uint32_t rankId{0};
    bool withMyRank{false};
    uint32_t localSize{0};
    uint32_t localId{0};
    std::vector<uint32_t> rankGroup;

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;

    ccu::Variable sliceSize;
    ccu::Variable inputSliceStride;
    ccu::Variable outputoffset;
    GroupOpSizeVars groupOpSize;

    std::vector<ccu::Variable> inputOffsets;

    ccu::Event event;

    uint16_t selfBit{0};
    uint16_t allBit{0};
};

CcuResult CcuAllToAllMesh2DieKernel(CcuKernelArg arg);

}

#endif
