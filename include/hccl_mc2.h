/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MC2_H
#define HCCL_MC2_H

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

constexpr uint32_t ALG_CONFIG_SIZE = 128;
struct HcclOpArgs {
    HcclDataType srcDataType;
    HcclDataType dstDataType;
    HcclReduceOp reduceType;
    uint64_t count;
    char algConfig[ALG_CONFIG_SIZE];
    CommEngine commEngine;
    uint64_t reverse;

    void Init() {
        srcDataType = HCCL_DATA_TYPE_FP16;
        dstDataType = HCCL_DATA_TYPE_FP16;
        reduceType = HCCL_REDUCE_SUM;
    }
}

extern HcclResult HcclKfcAllocOpArgs(void **opArgs);

extern HcclResult HcclKfcFreeOpArgs(void *opArgs);

extern HcclResult HcclKfcOpArgsSetSrcDataType(void *opArgs, uint8_t srcDataType);

extern HcclResult HcclKfcOpArgsSetDstDataType(void *opArgs, uint8_t dstDataType);

extern HcclResult HcclKfcOpArgsSetReduceType(void *opArgs, uint32_t reduceType);

extern HcclResult HcclKfcOpArgsSetCount(void *opArgs, uint64_t count);

extern HcclResult HcclKfcOpArgsSetAlgConfig(void *opArgs, char *algConfig);

extern HcclResult HcclKfcOpArgsSetCommEngine(void *opArgs, uint8_t commEngine);

extern HcclResult HcclCreateOpResCtx(HcclComm comm, uint8_t opType, void *opArgs, void **opResCtx);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif // HCCL_MC2_H