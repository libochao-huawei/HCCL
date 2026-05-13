/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D

#include <memory>
#include <map>
#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

#ifndef NHR_STEP_INFO_NS_DEFINED
#define NHR_STEP_INFO_NS_DEFINED
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
#endif

struct CcuKernelArgReduceScatterNHR1D: CcuKernelArgBase {
    uint64_t                                dimSize;
    uint64_t                                mySubCommRankId;
    uint64_t                                axisId;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<u32, u32>                      rank2ChannelIdx;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
    uint32_t                                axisSize;
};

struct ReduceScatterNHR1DMem2MemContext: CcuKernelCtxBase {
    const CcuKernelArgReduceScatterNHR1D *arg;

    uint32_t mySubCommRankId{0};
    uint64_t dimSize{0};
    uint32_t axisId{0};
    uint32_t localSize{0};
    uint32_t myRankIdx{0};
    HcclReduceOp reduceOp;
    HcclDataType dataType;
    HcclDataType outputDataType;
    uint32_t axisSize{0};

    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> token;
    ccu::Variable die0Size;
    ccu::Variable die1Size;
    ccu::Variable die0LastSliceSize;
    ccu::Variable die1LastSliceSize;
    ccu::Variable inputSliceStride;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable repeatNumVar;
    ccu::Variable repeatNumVarTemp;
    ccu::Variable sliceSize;

    ccu::Event event;

    ccu::Variable repeatInputOffset;
    ccu::Variable repeatOutputOffset;
    ccu::Variable isInputOutputEqual;
    ccu::Variable currentRankSliceOutputOffset;

    ccu::LocalAddr   localSrc;
    ccu::LocalAddr   localDst;
    ccu::RemoteAddr  remoteDst;
    ccu::Variable    isRepeatIter;
};

CcuResult CcuReduceScatterNHR1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_NHR_1D