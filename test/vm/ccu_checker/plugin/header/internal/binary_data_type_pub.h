/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BINRAY_DATA_TYPE_PUB_H
#define BINRAY_DATA_TYPE_PUB_H

#include <vector>
#include "hccl_types.h"
#include "ccu_microcode.h"
#include "task_meta_defs.h"

namespace HcclSim {

#define HCCLVM_TASK_FILE_MAGIC    0x48565444  // "HVTM"
#define HCCLVM_SYN_FILE_MAGIC     0x48564D44  // "HVMD"
#define HCCLVM_INSTR_FILE_MAGIC   0x434D4349  // "CMCI"

#pragma pack(push, 1)
struct FileHeader {
    uint32_t magic;
    uint16_t version{1};
    uint16_t header_size;
    uint32_t flags;
    uint32_t count;
    uint32_t checksum;
};

// ReduceScatterV AllGatherV使用
struct VDataDesTag {
    uint16_t dataType;   // 数据类型
    uint32_t count; // rank size
    uint64_t *displs;    // 每个rank的数据在sendBuf中的偏移量（单位为dataType）
    uint64_t *counts;    // 每个rank在sendBuf中的数据size，第i个元素表示需要向rank i发送/接受的数据量
};

struct All2AllDataDesTag {
    uint16_t sendType;   // 发送数据的数据类型
    uint16_t recvType;   // 接收数据的数据类型
    uint64_t sendCount;              // 发送数据量 (All2All)
    uint64_t recvCount;              // 接收数据量 (All2All)
    uint32_t count; // count = rankSize * rankSize
    uint64_t *sendCountMatrix;   // (All2AllVC) sendCountMatrix[i * ranksize + j] 代表rank i发送到rank j的count参数
};

struct ModelInfo {
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t root;
    uint32_t rank_size;
    uint16_t chip_type;
    uint16_t op_type;
    uint16_t reduce_op;
    uint16_t data_type;
    uint64_t data_count;
    VDataDesTag vDataDes;
    All2AllDataDesTag all2AllDataDes;
};

struct ChannelData {
    uint16_t channelId;
    uint8_t  srcDieId;
    uint8_t  dstDieId;
    uint32_t srcRank;
    uint32_t dstRank;
};

struct ChannelInfo {
    uint32_t count;
    ChannelData *data;
};

struct MemLayoutData {
    uint32_t rank_id;
    uint8_t  buffer_type;  // 类型
    uint8_t  reserved;
    uint64_t start_addr;    // 物理起始地址
    uint64_t size;         // 块大小
    uint64_t global_offset; // 该类型总偏移
};

struct MemLayoutInfo {
    uint32_t count;
    MemLayoutData *data;
};

struct MicrocodeInstrDesc {
    uint32_t rank_id;
    uint8_t  die_id;
    uint8_t  reserved;
    uint16_t count;
};

struct MicrocodeInstr {
    MicrocodeInstrDesc desc;
    hcomm::CcuRep::CcuInstr *data;
};

#pragma pack(pop)

struct ChannelInfoInner {
    uint32_t count;
    std::vector<ChannelData> data;
};

struct MemLayoutInfoInner {
    uint32_t count;
    std::vector<MemLayoutData> data;
};

// ReduceScatterV AllGatherV使用
struct VDataDesTagInner {
    uint16_t dataType;   // 数据类型
    uint32_t count{0}; // rank size
    std::vector<uint64_t> displs;    // 每个rank的数据在sendBuf中的偏移量（单位为dataType）
    std::vector<uint64_t> counts;    // 每个rank在sendBuf中的数据size，第i个元素表示需要向rank i发送/接受的数据量
};

struct All2AllDataDesTagInner {
    uint16_t sendType;   // 发送数据的数据类型
    uint16_t recvType;   // 接收数据的数据类型
    uint64_t sendCount;              // 发送数据量 (All2All)
    uint64_t recvCount;              // 接收数据量 (All2All)
    uint32_t count{0}; // count = rankSize * rankSize
    std::vector<uint64_t> sendCountMatrix;   // (All2AllVC) sendCountMatrix[i * ranksize + j] 代表rank i发送到rank j的count参数
};

struct ModelInfoCommInner {
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t root;
    uint32_t rank_size;
    uint16_t chip_type;
    uint16_t op_type;
    uint16_t reduce_op;
    uint16_t data_type;
    uint64_t data_count;
};

struct ModelInfoInner {
    ModelInfoCommInner comm;
    VDataDesTagInner vDataDes;
    All2AllDataDesTagInner all2AllDataDes;
};

struct HcclVmSynData {
    FileHeader header;
    ModelInfoInner  model_info;
    ChannelInfoInner channel_info;
    MemLayoutInfoInner memory_info;
};

struct MicrocodeInstrInner {
    MicrocodeInstrDesc desc;
    std::vector<hcomm::CcuRep::CcuInstr> data;
};

struct HcclVmInstrData {
    FileHeader header;
    std::vector<MicrocodeInstrInner> instr_data;
};

struct HcclVmTaskMetaData {
    FileHeader header;
    std::vector<HcclTaskMetaData> task_meta;
};

}
#endif