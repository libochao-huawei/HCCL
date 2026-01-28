/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_COMMON_H
#define SIM_COMMON_H
#include <iostream>
#include <vector>
#include "dtype_common.h"
#include "enum_factory.h"

MAKE_ENUM(BufferType, INPUT, OUTPUT, CCL, RESERVED)

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

using s8 = signed char;
using s16 = signed short;
using s32 = signed int;
using s64 = signed long long;

using RankId = uint32_t;
using PodId = uint32_t;
using SerId = uint32_t;
using PhyId = uint32_t;
constexpr uint64_t SIM_MEM_BLOCK_SIZE = 0x10000000000;  // 内存分配起始地址
constexpr uint64_t SIM_MEM_MASKER = 0xFFFFFF0000000000;
constexpr uint64_t SIZE_200MB = 200ULL * 1024 * 1024;  // CCL 200MB
constexpr uint32_t MAX_STREAM_COUNT = 1984;
constexpr uint32_t MAX_NOTIFY_COUNT = 8192;

using ChannelHandle = uint64_t;
using ThreadHandle = uint64_t;
using PhyDeviceId = uint32_t;
using ServerMeta = std::vector<PhyDeviceId>;
using SuperPodMeta = std::vector<ServerMeta>;
using TopoMeta = std::vector<SuperPodMeta>;

struct NpuPos {
    PodId superpodId;  // 超节点Id
    SerId serverId;    // ServerId
    PhyId phyId;       // Server内物理Id
};


struct MemBlock {
    BufferType bufferType;
    uint64_t startAddr;
    uint64_t size;
};

struct VDataDesTag {
    std::vector<u64> displs;    // 每个Rank的数据在buffer中的偏移量 单位为dataType
    std::vector<u64> counts;    // 每个Rank的数据大小 第i个表示需要向第i个Rank发送/接收的数据大小
    HcclDataType dataType;      // 数据类型
};

struct All2AllDataDesTag
{
    HcclDataType sendType;          // 发送数据类型
    HcclDataType recvType;          // 接收数据类型
    u64 sendCount;                  // All2All
    u64 recvCount;                  // All2All
    std::vector<u64> sendCounts;    // All2AllV 每个Rank发送的数据大小
    std::vector<u64> recvCounts;    // All2AllV 每个Rank接收的数据大小
    std::vector<u64> sdispls;       // All2AllV 每个Rank发送的数据在buffer中的偏移量 单位为dataType
    std::vector<u64> rdispls;       // All2AllV 每个Rank接收的数据在buffer中的偏移量 单位为dataType
    std::vector<u64> sendCountMatrix;   // All2AllVC sendCountMatrix[i * rankSize + j]表示第i个Rank向第j个Rank发送的数据大小
};


#endif