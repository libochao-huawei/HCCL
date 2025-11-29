/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_DEF_H
#define CHECKER_DEF_H

#include <vector>
#include "sim_common.h"
#include "hccl/base.h"
#include "enum_factory.h"

MAKE_ENUM(CheckerOpType,
    BROADCAST,
    ALLREDUCE,
    REDUCE,
    SEND,
    RECEIVE,
    ALLGATHER,
    REDUCE_SCATTER,
    ALLTOALLV,
    ALLTOALLVC,
    ALLTOALL,
    GATHER,
    SCATTER,
    BATCH_SEND_RECV,
    BATCH_PUT,
    BATCH_GET,
    ALLGATHER_V,
    REDUCE_SCATTER_V
);

enum CheckerOpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

MAKE_ENUM(CheckerReduceOp, REDUCE_SUM, REDUCE_PROD, REDUCE_MAX, REDUCE_MIN, REDUCE_RESERVED);

enum CheckerDataType {
    DATA_TYPE_INT8 = 0,    /**< int8 */
    DATA_TYPE_INT16 = 1,   /**< int16 */
    DATA_TYPE_INT32 = 2,   /**< int32 */
    DATA_TYPE_FP16 = 3,    /**< fp16 */
    DATA_TYPE_FP32 = 4,    /**< fp32 */
    DATA_TYPE_INT64 = 5,   /**< int64 */
    DATA_TYPE_UINT64 = 6,  /**< uint64 */
    DATA_TYPE_UINT8 = 7,   /**< uint8 */
    DATA_TYPE_UINT16 = 8,  /**< uint16 */
    DATA_TYPE_UINT32 = 9,  /**< uint32 */
    DATA_TYPE_FP64 = 10,   /**< fp64 */
    DATA_TYPE_BFP16 = 11,  /**< bfp16 */
    DATA_TYPE_INT128 = 12, /**< int128 */
    DATA_TYPE_HIF8 = 13,   /**< hif8 */
    DATA_TYPE_FP8E4M3 = 14,/**< fp8e4m3 */
    DATA_TYPE_FP8E5M2 = 15,/**< fp8e5m2 */
    DATA_TYPE_RESERVED     /**< reserved */
};

// 待补充
enum CheckerDevType {
    DEV_TYPE_910 = 0,
    DEV_TYPE_310P3 = 1, // PG
    DEV_TYPE_910B = 2,
    DEV_TYPE_310P1 = 3, // AG
    DEV_TYPE_910_93 = 4,
    DEV_TYPE_NOSOC = 5,
    DEV_TYPE_910_95 = 6,
    DEV_TYPE_COUNT = 7
};

typedef enum {
    CHECK_SEND = 0,
    CHECK_RECV = 1,
    CHECK_SEND_RECV_RESERVED
} CheckerSendRecvType;

typedef struct CheckerSendRecvItemDef {
    CheckerSendRecvType sendRecvType;
    void *buf;
    uint64_t count;
    CheckerDataType dataType;
    uint32_t remoteRank;
} CheckerSendRecvItem;

constexpr u32 CHECK_SIZE_TABLE[DATA_TYPE_RESERVED] = {sizeof(s8), sizeof(s16), sizeof(s32),
    2, sizeof(float), sizeof(s64), sizeof(u64), sizeof(u8), sizeof(u16), sizeof(u32), 8, 2, 16, 1, 1, 1};

struct CheckerOpParam {
    CheckerOpType opType;
    std::string tag;
    std::string algName;
    CheckerOpMode opMode;
    CheckerReduceOp reduceType = CheckerReduceOp::REDUCE_SUM;
    CheckerDevType devType = CheckerDevType::DEV_TYPE_910B;
    RankId root = -1;
    RankId dstRank = -1;
    RankId srcRank = -1;
    bool aicpuUnfoldMode = false;
    std::vector<std::vector<CheckerSendRecvItem>> allRanksSendRecvInfoVec; //batchsendrecv
    std::vector<CheckerDevType> devTypes; // 混合组网场景
    bool supportZeroCopy = false;
    bool isZeroCopy = false;
    u32 aiCoreLimit = 0;

    struct DataDesTag {
        u64 count;
        CheckerDataType dataType;
    } DataDes;

    // ReduceScatterV AllGatherV使用
    struct VDataDesTag {
        std::vector<u64> displs;
        std::vector<u64> counts;
        CheckerDataType dataType = CheckerDataType::DATA_TYPE_RESERVED;
    } VDataDes;

    // All2All系列使用
    struct All2AllDataDesTag {
        CheckerDataType sendType = CheckerDataType::DATA_TYPE_RESERVED;
        CheckerDataType recvType = CheckerDataType::DATA_TYPE_RESERVED;
        u64 sendCount;
        u64 recvCount;
        std::vector<u64> sendCounts;
        std::vector<u64> recvCounts;
        std::vector<u64> sdispls;
        std::vector<u64> rdispls;
        std::vector<u64> sendCountMatrix;
    } All2AllDataDes;
};

#endif