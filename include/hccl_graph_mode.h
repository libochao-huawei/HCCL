/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_GRAHP_MODE_H
#define HCCL_GRAHP_MODE_H

#include <cstdint>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <acl/acl.h>


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
constexpr uint32_t RES_PACK_TAG_LENGTH = 255;
// 图模式编译阶段资源计算入参
struct OpParamGraphMode
{
    /* data */
};

// 图模式编译阶段申请资源
struct ResResponseGraphMode {
    uint64_t opMemSize = 0;  // 额外申请的scratch数量（不包括cclBuff）
    uint64_t streamNum = 0;  // 除用户流以外，额外申请的流（不包括算子device展开申请的流）
    uint64_t taskNum = 0;    // task数量，一般为前同步 + kernel + 后同步
    uint64_t aivCoreNum = 0;
};

// 图模式执行阶段传入的资源
struct ResPackGraphMode {
    char tag[RES_PACK_TAG_LENGTH];
    std::vector<aclrtStream> streams;
    void* scratchMemAddr;
    uint64_t scratchMemSize;
};

/**
 * @brief calculate resource in online graph mode.
 *
 * @param opParam A struct contains operator running infomation to determine resouces.
 * @param resResponse A struct contain the resouces size the operator needs.
 * @return HcclResult
 */
HcclResult HcclCalcOpResOnlineGraphMode(OpParamGraphMode *opParam, ResResponseGraphMode *resResponse);

/**
 * @brief AllGather operator in offline graph mode.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param sendCount An integer(u64) identifying the number of the input data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, int64,
 * uint8, uint16, uint32, uint64, float16, float32, float64, bfp16.
 * @param group A pointer identifying the name of communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @param resPack A struct contains resource the operator demands.
 * @return HcclResult
 */
HcclResult HcclCalcOpResOfflineGraphMode(OpParamGraphMode *opParam, ResResponseGraphMode *resResponse);

/**
 * @brief AllGather operator in graph mode.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param sendCount An integer(u64) identifying the number of the input data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, int64,
 * uint8, uint16, uint32, uint64, float16, float32, float64, bfp16.
 * @param group A pointer identifying the name of communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @param resPack A struct contains resource the operator demands.
 * @return HcclResult
 */
extern HcclResult HcclAllGatherGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, const char* group, aclrtStream stream,
                                  const ResPackGraphMode &resPack);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_GRAHP_MODE_H