/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_MUTILJETTY_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_MUTILJETTY_1D_MEM2MEM_H

#include <vector>
#include <map>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};

namespace ops_hccl {

struct CcuKernelArgReduceScatterNhrMultiJettyMem2Mem1D : CcuKernelArgBase {
    uint64_t                             dimSize;
    uint32_t                             rankId;
    uint32_t                             portSize;
    OpParam                              opParam;
    std::vector<NHRStepInfo>             stepInfoVector;
    std::map<u32, u32>                   rank2ChannelIdx;
    std::vector<std::vector<uint32_t>>   subCommRanks;
};

struct ReduceScatterNhrMultiJettyMem2Mem1DContext : CcuKernelCtxBase {
    const CcuKernelArgReduceScatterNhrMultiJettyMem2Mem1D *arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    std::vector<CcuVariable> input;
    CcuVariable output;
    std::vector<CcuVariable> token;
    CcuVariable sliceSize;
    CcuVariable inputSliceStride;
    CcuVariable inputRepeatStride;
    CcuVariable outputRepeatStride;
    CcuVariable sliceOneJettySize;
    CcuVariable sliceLastJettySize;
    CcuVariable repeatNumVar;
    CcuVariable repeatNumVarTemp;
    CcuVariable flag;

    ccu::LocalAddr  localSrc;
    ccu::LocalAddr  localDst;
    ccu::RemoteAddr remoteDst;

    std::vector<CcuEvent> jettyEvents;
    CcuEvent event;
};

CcuResult CcuReduceScatterNhrMultiJettyMem2Mem1DKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D_MUTIL_JETTY_MEM2MEM_H
