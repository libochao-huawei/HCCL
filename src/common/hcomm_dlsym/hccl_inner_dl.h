/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INNER_DL_H
#define HCCL_INNER_DL_H

#include "hccl_inner.h"   // 原始头文件，包含所有类型和声明

#ifdef __cplusplus
extern "C" {
#endif

HcclResult __attribute__((weak)) HcclAllReduceInner(void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType,
                                         HcclReduceOp op, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclBroadcastInner(void* buf, uint64_t count, HcclDataType dataType, uint32_t root,
                                         HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclReduceScatterInner(void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                                             HcclReduceOp op, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclReduceScatterVInner(void* sendBuf, const void* sendCounts, const void* sendDispls,
                                              void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                                              HcclReduceOp op, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclScatterInner(void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                                       uint32_t root, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclAllGatherInner(void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType,
                                         HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclAllGatherVInner(void* sendBuf, uint64_t sendCount, void* recvBuf,
                                          const void* recvCounts, const void* recvDispls,
                                          HcclDataType dataType, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclSendInner(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                    HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclRecvInner(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                                    HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclAlltoAllVCInner(const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType,
                                          const void* recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclAlltoAllVInner(const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType,
                                         const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType recvType,
                                         HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclAlltoAllInner(const void* sendBuf, uint64_t sendCount, HcclDataType sendType,
                                        const void* recvBuf, uint64_t recvCount, HcclDataType recvType,
                                        HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclReduceInner(void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType,
                                      HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclBatchSendRecvInner(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);
HcclResult __attribute__((weak)) HcclCreateOpResCtxInner(HcclComm comm, uint8_t opType, HcclDataType srcDataType, HcclDataType dstDataType,
                                              HcclReduceOp reduceType, uint64_t count, char* algConfig, uint32_t commEngine, void** opResCtx);

// 查询函数声明
bool HcommIsSupportHcclAllReduceInner(void);
bool HcommIsSupportHcclBroadcastInner(void);
bool HcommIsSupportHcclReduceScatterInner(void);
bool HcommIsSupportHcclReduceScatterVInner(void);
bool HcommIsSupportHcclScatterInner(void);
bool HcommIsSupportHcclAllGatherInner(void);
bool HcommIsSupportHcclAllGatherVInner(void);
bool HcommIsSupportHcclSendInner(void);
bool HcommIsSupportHcclRecvInner(void);
bool HcommIsSupportHcclAlltoAllVCInner(void);
bool HcommIsSupportHcclAlltoAllVInner(void);
bool HcommIsSupportHcclAlltoAllInner(void);
bool HcommIsSupportHcclReduceInner(void);
bool HcommIsSupportHcclBatchSendRecvInner(void);
bool HcommIsSupportHcclCreateOpResCtxInner(void);

// 动态库管理接口
void HcclInnerDlInit(void* libHcommHandle);
void HcclInnerDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_INNER_DL_H