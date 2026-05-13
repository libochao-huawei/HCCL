/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgReduceScatterMesh1DMem2Mem: CcuKernelArgBase{
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct ReduceScatterMesh1DMem2MemContext: CcuKernelCtxBase {
    const CcuKernelArgReduceScatterMesh1DMem2Mem *arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> scratch;
    std::vector<ccu::Variable> token;
    ccu::Variable output;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable repeatNum;
    ccu::Variable flag;
    GroupOpSizeVars goSize;

    uint16_t selfBit;
    uint16_t allBit;

    ccu::LocalAddr  myInput;
    std::vector<ccu::RemoteAddr> remoteInput;
    std::vector<ccu::LocalAddr>  scratchMem;
    ccu::Event event;
};

CcuResult CcuReduceScatterMesh1DMem2MemKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_MEM2MEM