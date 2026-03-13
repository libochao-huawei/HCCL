/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: checker所需的数据结构定义
 * Author: huangweihao
 * Create: 2025-04-30
 */
#ifndef HCCLV1_CHECKER_DEF_H
#define HCCLV1_CHECKER_DEF_H

#include "llt_common.h"
#include <vector>
#include <set>
#ifndef HCCL_ALG_ANALYZER_DAVID
#include "op_context.h"
#endif
#include "enum_factory.h"

namespace checker {
// 待补充
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
    REDUCE_SCATTER_V,
    BATCH_WRITE,
    ALL,
    MAX
);

enum CheckerOpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

MAKE_ENUM(CheckerReduceOp, REDUCE_SUM, REDUCE_PROD, REDUCE_MAX, REDUCE_MIN, REDUCE_RESERVED);

// 待补充
enum CheckerDataType {
    DATA_TYPE_INT8 = 0,    /**< int8 */
    DATA_TYPE_INT16 = 1,   /**< int16 */
    DATA_TYPE_INT32 = 2,   /**< int32 */
    DATA_TYPE_FP16 = 3,    /**< fp16 */
    DATA_TYPE_FP32 = 4,    /**< fp32 */
    DATA_TYPE_INT64 = 5,    /**< int64 */
    DATA_TYPE_UINT64 = 6,    /**< uint64 */
    DATA_TYPE_UINT8 = 7,    /**< uint8 */
    DATA_TYPE_UINT16 = 8,   /**< uint16 */
    DATA_TYPE_UINT32 = 9,   /**< uint32 */
    DATA_TYPE_FP64 = 10,    /**< fp64 */
    DATA_TYPE_BFP16 = 11,    /**< bfp16 */
    DATA_TYPE_INT128 = 12,   /**< int128 */
    DATA_TYPE_HIF8 = 13,     /**< hif8 */
    DATA_TYPE_FP8E4M3 = 14,  /**< fp8e4m3 */
    DATA_TYPE_FP8E5M2 = 15,  /**< fp8e5m2 */
    DATA_TYPE_RESERVED       /**< reserved */
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
    DEV_TYPE_HF = 7,
    DEV_TYPE_COUNT = 8
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
    CheckerDevType devtype = CheckerDevType::DEV_TYPE_910B;
    bool is310P3V = false;  // 仅当310PV卡的时候，设置为1
    RankId root = -1;
    RankId dstRank = -1;
    RankId srcRank = -1;
#ifndef HCCL_ALG_ANALYZER_DAVID
    hccl::AlgOpContext algOpContext;
#endif
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
        CheckerDataType dataType;
    } VDataDes;

    // All2All系列使用
    struct All2AllDataDesTag {
        CheckerDataType sendType;
        CheckerDataType recvType;
        u64 sendCount;
        u64 recvCount;
        std::vector<u64> sendCounts;
        std::vector<u64> recvCounts;
        std::vector<u64> sdispls;
        std::vector<u64> rdispls;
        std::vector<u64> sendCountMatrix;
    } All2AllDataDes;

    struct {
        u32 itemNum;
        u32 queueNum;
        u32 queueIdx;
    } BatchWriteDataDes;
};

} // namespace hccl

#endif