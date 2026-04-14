/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D

#include <vector>
#include <ios>
// #include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint64_t RS_MAX_RANK_SIZE         = 128;

struct CcuKernelArgReduceScatterMesh1D: CcuKernelArgBase{
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct ReduceScatterContext: CcuKernelCtxBase {
    const CcuKernelArgReduceScatterMesh1D *arg;
    
    uint64_t rankSize{0};
    uint32_t rankId{0};
    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;
    std::vector<ChannelHandle> channels;
    std::vector<CcuVariable> input;
    CcuVariable output;
    std::vector<CcuVariable> token;
    CcuVariable offset;
    GroupOpSizeVars goSize;

    // LoopGroupConfig  moConfig;
    // LoopGroupResource moRes;
    // bool resourceAllocated;

    CcuLoopHandle reduceLoops[2];
    bool loopRegistered;

    // Loop body 中的外部 LocalAddr（每个 loop index 各两组）
    CcuLocalAddr loopDst[2];
    CcuLocalAddr loopSrc[2];
    CcuLocalAddr loopScratch[2][RS_MAX_RANK_SIZE];
    CcuVariable  loopLen[2];
    CcuVariable  loopLenExp[2];
};

HcclResult CcuReduceScatterMesh1DKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D