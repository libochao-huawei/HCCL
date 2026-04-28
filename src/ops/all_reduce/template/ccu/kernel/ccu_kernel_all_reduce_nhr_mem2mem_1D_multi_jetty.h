/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_NHR_1D_MEM2MEM_MULTI_JETTY_H
#define HCCL_CCU_KERNEL_ALL_REDUCE_NHR_1D_MEM2MEM_MULTI_JETTY_H

#include <vector>
#include <map>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

enum class XnId {
    OUTPUT = 0,
    TOKEN
};

enum class SignalBit {
    PRE_SYNC_OUTPUT = 0,
    PRE_SYNC_TOKEN,
    READY_TO_RECV_RS,
    READY_TO_RECV_AG,
    SEND_DONE_RS,
    SEND_DONE_AG,
    POST_SYNC
};

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

struct CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty: CcuKernelArgBase {
    uint32_t                             rankSize;
    uint32_t                             rankId;
    uint32_t                             portSize;
    OpParam                              opParam;
    std::vector<NHRStepInfo>             algStepInfoList;
    std::map<u32, u32>                   channelIdxMap;
    std::vector<std::vector<uint32_t>>   subCommRanks;
};

struct AllReduceNhrMem2Mem1DMultiJettyContext: CcuKernelCtxBase {
    const CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    CcuVariable inputAddr;
    std::vector<CcuVariable> outputAddrs;
    std::vector<CcuVariable> outputTokens;
    CcuVariable isInplace;
    CcuVariable dataSizePerRank;
    CcuVariable dataSizePerPort;
    CcuVariable lastRankSliceSize;
    CcuVariable lastPortSliceSize;
    std::vector<CcuVariable> sliceOffset;
    GroupOpSizeVars localCopyGoSize;
    GroupOpSizeVars localCopyGoSizeLastSlice;

    ccu::LocalAddr  localInput;
    ccu::LocalAddr  localOutput;
    ccu::RemoteAddr remoteOutput;

    std::vector<CcuEvent> events;
};

CcuResult CcuAllReduceNhrMem2Mem1DMultiJettyKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_REDUCE_NHR_1D_MEM2MEM_MULTI_JETTY_H
