/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_NHR1D_MEM2MEM
#define HCCL_CCU_KERNEL_ALL_GATHER_NHR1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllGatherNHR1D : CcuKernelArgBase {
    uint64_t                                dimSize;
    uint64_t                                mySubCommRankId;
    uint64_t                                axisId;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<u32, u32>                      rank2ChannelIdx;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
    uint32_t                                axisSize;
};

struct AllGatherNHR1DMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgAllGatherNHR1D *arg;

    uint64_t localSize;
    uint64_t myRankIdx;
    CcuVariable input;
    std::vector<CcuVariable> output;
    std::vector<CcuVariable> token;
    CcuVariable die0Size;
    CcuVariable die1Size;
    CcuVariable repeatNum;
    CcuVariable inputSliceStride;
    CcuVariable outputSliceStride;
    CcuVariable inputRepeatStride;
    CcuVariable outputRepeatStride;
    CcuVariable isInputOutputEqual;
    CcuVariable die0LastSize;
    CcuVariable die1LastSize;
    CcuEvent localEvent;
    CcuVariable repeatTimeflag;
    std::vector<CcuVariable> outputSliceOffset;
    CcuVariable myrankInputSliceOffset;
    CcuLocalAddr srcMem;
    CcuRemoteAddr dstMem;
    CcuLocalAddr localDst;
    CcuVariable constVar1;
};

CcuResult CcuAllGatherNHR1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_NHR1D_MEM2MEM
