/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H

#include <vector>
#include <ios>
#include "utils.h"
// #include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAlltoAllMesh1D: CcuKernelArgBase{
    uint64_t                                rankSize;
    uint32_t                                rankId;
    bool                                    loadFromMem;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct AlltoAllMesh1DContext: CcuKernelCtxBase {
    const CcuKernelArgAlltoAllMesh1D *arg;
    
    // uint64_t rankSize{0};
    // uint32_t rankId{0};
    HcclDataType dataType;
    HcclDataType outputDataType;
    ccu::CcuVariable repeatNum;
    // std::vector<ChannelHandle> channels;
    std::vector<ccu::CcuVariable> input;
    std::vector<ccu::CcuVariable> output;
    std::vector<ccu::CcuVariable> token;
    ccu::CcuVariable sliceSize;
    ccu::CcuVariable srcStride;
    ccu::CcuVariable srcOffset;
    ccu::CcuVariable dstOffset;
    GroupOpSizeVars goSize;

    ccu::LocalAddr                   myDst;
    std::vector<ccu::RemoteAddr>     dstAddr;
    std::vector<ccu::LocalAddr>      srcAddr;
    std::vector<ccu::CcuVariable>         dstToken;
    std::vector<ccu::CcuVariable>         srcToken;
    ccu::CcuEvent event;
    ccu::CcuVariable flag; // 用以判断是否是第一次重复
    bool loadFromMem = false;
};

CcuResult CcuAlltoAllMesh1DKernel(CcuKernelArg arg);

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H
