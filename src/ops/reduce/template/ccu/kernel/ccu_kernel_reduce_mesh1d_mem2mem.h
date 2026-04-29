/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
constexpr uint64_t LOCAL_COPY_MS = 8;

struct CcuKernelArgReduceMesh1DMem2Mem: CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    uint32_t                                rootId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct GroupReduceMesh1DMem2MemVar {
    CcuLocalAddr src[2];
    CcuLocalAddr dst[2];
    CcuVariable  len[2];
};

struct ReduceMesh1DMem2MemContext: CcuKernelCtxBase {
    const CcuKernelArgReduceMesh1DMem2Mem *arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    std::vector<CcuVariable> input;
    std::vector<CcuVariable> output;
    std::vector<CcuVariable> token;

    CcuVariable inputRepeatStride;
    CcuVariable outputRepeatStride;
    CcuVariable normalSliceSize;
    CcuVariable lastSliceSize;
    CcuVariable repeatNumVar;
    CcuVariable flag;
    CcuVariable isInputOutputEqual;
    std::vector<CcuVariable> chunkSize;
    CcuVariable chunkOffset;

    CcuEvent event;

    CcuLocalAddr myInputAddr;
    CcuRemoteAddr remoteInputAddr;
    CcuLocalAddr dstAddr;

    GroupOpSizeVars localGoSize;
};

CcuResult CcuReduceMesh1DMem2MemKernel(CcuKernelArg arg);

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM
