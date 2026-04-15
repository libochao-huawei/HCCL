/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_ALLTOALLV_AICPU_COMMON_H
#define OPS_HCCL_ALLTOALLV_AICPU_COMMON_H

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <acl/acl_rt.h>
#include <vector>

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t MAX_RANK_SIZE = 8;

constexpr uint32_t AICPU_CONTROL_NOTIFY_NUM = 2;

enum DeviceType {
    DEVICE_TYPE_A2 = 0,
    DEVICE_TYPE_A3 = 1,
    DEVICE_TYPE_A5 = 2,
    DEVICE_TYPE_950 = 3,
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct AlgResourceCtx {
    ThreadHandle mainThread;
    ThreadHandle cpuThreadOnAicpu;
    std::vector<ThreadHandle> slaveThreads;
    std::vector<ChannelHandle> channelHandles;
    CommBuffer cclBuffer;
    CommBuffer inputBuffer;
    CommBuffer outputBuffer;
    uint32_t notifyNumOnMainThread;
    std::vector<uint32_t> notifyNumPerThread;
};

struct AlltoAllVDataDes {
    uint64_t* sendCounts;
    uint64_t* recvCounts;
    uint64_t* sdispls;
    uint64_t* rdispls;
    HcclDataType sendType;
    HcclDataType recvType;
};

struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    void* inputPtr = nullptr;
    void* outputPtr = nullptr;
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    uint64_t count = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    DeviceType devType;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    ThreadHandle cpuThread;
    ThreadHandle aicpuThreadOnCpu;
    AlgResourceCtx* resCtx = nullptr;
    AlltoAllVDataDes all2AllVDataDes;
    void* varData = nullptr;
    uint64_t varMemSize = 0;
};

constexpr uint64_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1
};

constexpr uint32_t ALL_TO_ALL_V_VECTOR_NUM = 4;

#endif