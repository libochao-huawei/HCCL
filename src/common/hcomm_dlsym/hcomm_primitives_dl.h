/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PRIMITIVES_DL_H
#define HCOMM_PRIMITIVES_DL_H

#include "dlsym_common.h"
#include "hcomm_primitives.h"   // 原头文件，包含所有类型和定义
#include "hccl_types.h"

/* 8.5.0 桩: HcclCommSymWindow (来自 hccl_types.h) */
#if CANN_VERSION_NUM < 90000000
typedef void *HcclCommSymWindow;
typedef enum {
    HCOMM_TRANSFER_TYPE_INVALID = -1,
    HCOMM_TRANSFER_TYPE_WRITE = 0,
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE = 1,
    HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3,
    HCOMM_TRANSFER_TYPE_READ = 4,
    HCOMM_TRANSFER_TYPE_READ_REDUCE = 5,
    HCOMM_TRANSFER_TYPE_NOTIFY_RECORD = 6,
    HCOMM_TRANSFER_TYPE_NOTIFY_WAIT = 7,
    HCOMM_TRANSFER_TYPE_NOTIFY_WAIT_WITH_DEFAULT_TIMEOUT = 8
} HcommTransferType;

typedef struct {
    HcommTransferType transType;
    uint8_t reserved[4];
    union {
        uint8_t raws[56];
        struct {
            uint64_t len;
            void *dst;
            void *src;
        } write;
        struct {
            uint64_t len;
            void *dst;
            void *src;
        } read;
        struct {
            uint64_t count;
            void *dst;
            void *src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
        } reduce;
        struct {
            uint32_t notifyIdx;
        } notifyRecord;
        struct {
            uint32_t notifyIdx;
            uint32_t timeOut;
        } notifyWait;
        struct {
            uint32_t notifyIdx;
        } notifyWaitWithDefaultTimeout;
        struct {
            uint64_t len;
            void *dst;
            void *src;
            uint32_t notifyIdx;
        } writeWithNotify;
        struct {
            uint64_t count;
            void *dst;
            void *src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
            uint32_t notifyIdx;
        } writeReduceWithNotify;
    } transferInfo;
} HcommBatchTransferDesc;
#endif

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(int32_t, HcommThreadSynchronize, ThreadHandle thread);
DECL_WEAK_FUNC(int32_t, HcommSendRequest, uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId);
DECL_WEAK_FUNC(int32_t, HcommWaitResponse, uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId);
DECL_WEAK_FUNC(HcclResult, HcommThreadJoin, ThreadHandle thread, uint32_t timeout);
DECL_WEAK_FUNC(int32_t, HcommBatchTransferOnThread, ThreadHandle thread, ChannelHandle channel,
    HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum);

void HcommPrimitivesDlInit(void* libHcommHandle);  // 本模块独立初始化

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PRIMITIVES_DL_H
