/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AIV_KERNEL_OMNI_H_
#define HCCL_AIV_KERNEL_OMNI_H_

#include <cstdint>

namespace ops_hccl {
constexpr uint32_t AIV_OMNI_MAX_SLICE_CNT = 16;

enum AivOmniOpType : uint32_t {
    AIV_OMNI_OP_LOCAL_COPY = 0,
    AIV_OMNI_OP_LOCAL_REDUCE = 1,
    AIV_OMNI_OP_SEND_RECV_WRITE = 2,
    AIV_OMNI_OP_SEND_WRITE = 3,
    AIV_OMNI_OP_RECV_WRITE = 4,
    AIV_OMNI_OP_SEND_RECV_WRITE_REDUCE = 5,
    AIV_OMNI_OP_SEND_WRITE_REDUCE = 6,
    AIV_OMNI_OP_RECV_WRITE_REDUCE = 7,
    AIV_OMNI_OP_SEND_RECV_READ = 8,
    AIV_OMNI_OP_SEND_READ = 9,
    AIV_OMNI_OP_RECV_READ = 10,
    AIV_OMNI_OP_SEND_RECV_READ_REDUCE = 11,
    AIV_OMNI_OP_SEND_READ_REDUCE = 12,
    AIV_OMNI_OP_RECV_READ_REDUCE = 13,
    AIV_OMNI_OP_GROUP_BROAD_CAST = 14,
    AIV_OMNI_OP_GROUP_REDUCE = 15,
};

struct AivOmniSliceInfo {
    uint64_t sliceType = 0;
    uint64_t sliceIdx = 0;
    uint64_t remoteRank = 0;
};

struct AivOmniSendRecvInfo {
    uint32_t opType = 0;
    uint32_t inputDataType = 0;
    uint32_t outputDataType = 0;
    uint32_t reduceType = 0;
    uint32_t srcSliceNum = 0;
    uint32_t dstSliceNum = 0;
    uint64_t sliceNum = 0;
    uint64_t linkType = 0;
    uint64_t threadIdx = 0;
    AivOmniSliceInfo srcSliceInfo[AIV_OMNI_MAX_SLICE_CNT] = {};
    AivOmniSliceInfo dstSliceInfo[AIV_OMNI_MAX_SLICE_CNT] = {};
};

struct AivOmniInfoHeader {
    uint64_t infoNum = 0;
};
} // namespace ops_hccl

#endif // HCCL_AIV_KERNEL_OMNI_H_
