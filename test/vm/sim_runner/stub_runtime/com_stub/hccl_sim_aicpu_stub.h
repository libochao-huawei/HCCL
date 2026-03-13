/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim aicpu stub
 */

#ifndef HCCL_SIM_AICPU_STUB_H
#define HCCL_SIM_AICPU_STUB_H
#include <iostream>
#include <string>
#include "hccl_common.h"
#include "aicpu_operator_pub.h"

using RankId = u32;

constexpr u32 MAX_OP_TAG_LEN = 191; // 最大的tag 长度, 和对外接口保持一致
constexpr u32 MAX_NAME_LEN   = 64;
constexpr u32 KERNEL_PARAM_NAME_SIZE = 32;

constexpr int HCCL_AICPU_COMM_LITE_SIZE = 184;
constexpr int HCCL_AICPU_OP_LITE_SIZE = 216;
constexpr int HCCL_HD_COMM_PARAM_SIZE = 40;
constexpr int HCCL_DEV_CONFIG_LITE_SIZE = 8;

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInit(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunch(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvGroupLaunch(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuKfcSrvLaunch(void *args[]);
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInitV2(void *args);
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunchV2(void *args);
__attribute__((visibility("default"))) uint32_t HcclKernelEntrance(void *args);
__attribute__((visibility("default"))) uint32_t HcclUpdateCommKernelEntrance(void *args);
}

struct OpTilingDataStub {
    char tag[128];
    char newTag[256];
    char algName[128];
    u32 index; // 集合通信算子在通信域内的编号，aicpu侧使用时区分算子类型(bsr、sendrecv、其他)单独计数
    u64 algType;
    u8 floatOverflowMode;
    u8 dumpDebug;
    u8 debugMode;
    u8 workflowMode;
    u64 inputPtr;
    u64 outputPtr;
    u8 reduceType;  // HcclReduceOp ::HCCL_REDUCE_RESERVED
    u8 syncMode;    // SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    RankId root = INVALID_VALUE_RANKID;
    RankId dstRank = 0;
    RankId srcRank = 0;
    u8 opType; // HcclCMDType::HCCL_CMD_INVALID;
    u8 inplaceSupportRetry;
    u8 retryEnable;
    u8 inPlaceSupportRetryStatus;
    u8 isInplacePreSync;
    u8 isPostSync;
    u8 isZeroCopy = 0;
    u64 version = 0;
    s32 userStreamId;
    u32 ahcConfInfo[TOP_HIERARCHICAL_CONF_SIZE] = {0};

    /******************可变长度数据区，如需新增字段请在这之前增加*******************/
    u64 length;   // 可变长度数据区长度
    u64 customDataLength;  // 用户自定义预留可变长度数据区长度，预期在aicpu侧做数据块校验
    u8 isCapture = 0;  // 算子是否aclgraph模式

    /* 不同算子，长度不同，依据opType决定选择使用
    * (1)batchsendrcv
    * struct {
    *     u32 itemNum;
    *     HcclSendRecvItem orderedList[itemNum];
    * } OpTilingBatchSendRecvDataDes;
    * (2)alltoallv
    * struct {
    *     u8 sendType;  // HcclDataType
    *     u8 recvType;  // HcclDataType
    *     u32 rankSize;
    *     u64 sendCounts[rankSize];
    *     u64 recvCounts[rankSize];
    *     u64 sdispls[rankSize];
    *     u64 rdispls[rankSize];
    * };
    * (3)alltoallvc
    * struct {
    *     u8 sendType;  // HcclDataType
    *     u8 recvType;  // HcclDataType
    *     u32 rankSize;
    *     u64 sendCountMatrix[rankSize * rankSize];
    * };
    *  (4)alltoall
    * struct {
    *     u8 sendType;  // HcclDataType
    *     u8 recvType;  // HcclDataType
    *     u64 sendCount;
    *     u64 recvCount;
    * };
    *  (5)other operators
    * struct {
    *     u64 count;
    *     u8 datatype;  // HcclDataType
    * }; */
};

struct HcclKernelParamLiteStub {
    uint64_t                  binaryResAddr{0};
    uint64_t                  binaryResSize{0};
    uint8_t                   comm[HCCL_AICPU_COMM_LITE_SIZE]{0};
    uint8_t                   op[HCCL_AICPU_OP_LITE_SIZE]{0};
    char                      algName[MAX_NAME_LEN]{0};
    bool                      needUpdateRes{false};
    bool                      oneSidedComm{false};
    char                      opTag[MAX_OP_TAG_LEN]{0};
    uint8_t                   kfcControlTransferH2DParams[HCCL_HD_COMM_PARAM_SIZE];
    uint8_t                   kfcControlTransferD2HParams[HCCL_HD_COMM_PARAM_SIZE];
    uint8_t                   envConfig[HCCL_DEV_CONFIG_LITE_SIZE];
};
 
struct HcclKernelLaunchParamStub {
    HcclKernelParamLiteStub kernel;
    char                soName[32]                         = "libccl_kernel.so";
    char                kernelName[KERNEL_PARAM_NAME_SIZE] = "HcclKernelEntrance";
    char                opName[32]                         = "LoadWithOpBasedMode";
};
 
struct KFCTaskCommStub {
    u64 context;     // HCCL通信context
    u64 tilingData;  // 通信
};

#endif // HCCL_SIM_AICPU_STUB_H