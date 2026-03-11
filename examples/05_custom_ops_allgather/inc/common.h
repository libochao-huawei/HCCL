/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_COMMON_H
#define OPS_HCCL_CUSTOM_COMMON_H

#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_primitives.h"
#include "acl/acl_rt.h"
#include "log.h"
#include "extra_args.h"
#include <vector>

namespace ops_hccl_allgather {

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;

constexpr uint64_t AIV_TAG_BUFF_LEN = 2 * 1024 * 1024; // 2MB

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct AlgResourceCtx {
    CommBuffer cclMem;        // Scratch buffer
    CommBuffer aivCommInfo;   // AIV communication info buffer
    
    // Channels to other ranks (simplified for mesh_1d: just a list)
    std::vector<ChannelHandle> channels;
    std::vector<CommBuffer> remoteBuffers; // Remote buffers corresponding to channels
};

struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    
    // AIV Kernel Arguments (matching EXTERN_KERNEL_ARGS_DEF_V2)
    void* buffIn = nullptr; // cclMem address
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t xRankSize = 0;
    uint64_t yRankSize = 0;
    uint64_t zRankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tagId = 0; // Maps to 'tag' in kernel
    
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    
    bool isOpBase = false;
    
    void* headCountMem = nullptr;
    void* tailCountMem = nullptr;
    void* addOneMem = nullptr;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
    
    ExtraArgs extraArgs;

    // Resource Context
    AlgResourceCtx* resCtx = nullptr;
};

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

}

#endif // OPS_HCCL_CUSTOM_COMMON_H
