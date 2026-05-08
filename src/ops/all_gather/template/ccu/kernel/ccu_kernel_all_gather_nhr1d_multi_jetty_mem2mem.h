/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM
#define HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem : CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    uint32_t                                jettyNum;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<uint32_t, uint32_t>            rank2ChannelIdx;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct AllGatherNHR1DMultiJettyMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *arg;

    uint64_t                                localSize;
    uint64_t                                myRankIdx;

    CcuVariable                             input;
    std::vector<CcuVariable>                output;
    std::vector<CcuVariable>                token;
    CcuVariable                             sliceSize;
    CcuVariable                             sliceSizePerJetty;
    CcuVariable                             lastSliceSizePerJetty;
    CcuVariable                             repeatNumInv;
    CcuVariable                             inputSliceStride;
    CcuVariable                             outputSliceStride;
    CcuVariable                             inputRepeatStride;
    CcuVariable                             outputRepeatStride;
    CcuVariable                             isInputOutputEqual;
    GroupOpSizeVars                         groupOpSize;
    CcuEvent                                event;
    std::vector<CcuVariable>                outputSliceOffset;
    CcuVariable                             constVar1;
    CcuLocalAddr                            srcMem;
    CcuRemoteAddr                           dstMem;
    CcuLocalAddr                            myDstMem;
    CcuVariable                             repeatTimeflag;
    CcuVariable                             tmpCopyRepeatNumInv;
};

CcuResult CcuAllGatherNHR1DMultiJettyMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM
