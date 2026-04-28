/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_ONE_SHOT_H_
#define HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_ONE_SHOT_H_

#include <vector>
#include <ios>
#include "log.h"
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_alg_base.h"
#include "alg_param.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

struct CcuKernelArgAllReduceMesh1DOneShot: CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct AllReduceMesh1DOneShotContext: CcuKernelCtxBase {
    const CcuKernelArgAllReduceMesh1DOneShot *arg;
    
    // uint64_t rankSize{0};
    // uint32_t rankId{0};
    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;
    // std::vector<ChannelHandle> channels;
    std::vector<CcuVariable> input;
    CcuVariable output;
    std::vector<CcuVariable> token;
    GroupOpSizeVars goSize;

    // LoopGroupConfig  moConfig;
    // LoopGroupResource moRes;
    // bool resourceAllocated;

    //CcuLoop reduceLoops[2];
    //bool loopRegistered;

    // Loop body 中的外部 LocalAddr（每个 loop index 各两组）
    //ccu::LocalAddr loopDst[2];
    //ccu::LocalAddr loopSrc[2];
    //ccu::LocalAddr loopScratch[2][RS_MAX_RANK_SIZE];
    //CcuVariable  loopLen[2];
    //CcuVariable  loopLenExp[2];
};

CcuResult CcuAllReduceMesh1DOneShotKernel(CcuKernelArg arg);

} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_