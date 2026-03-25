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

#include "hcomm_primitives.h"   // 原头文件，包含所有类型和定义

#ifdef __cplusplus
extern "C" {
#endif

int32_t __attribute__((weak)) HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len);
int32_t __attribute__((weak)) HcommLocalReduceOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t count,
                                            HcommDataType dataType, HcommReduceOp reduceOp);
int32_t __attribute__((weak)) HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx);
int32_t __attribute__((weak)) HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut);
int32_t __attribute__((weak)) HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId);
int32_t __attribute__((weak)) HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut);
int32_t __attribute__((weak)) HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len);
int32_t __attribute__((weak)) HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                            uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
int32_t __attribute__((weak)) HcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                uint64_t len, uint32_t remoteNotifyIdx);
int32_t __attribute__((weak)) HcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                      uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp,
                                                      uint32_t remoteNotifyIdx);
int32_t __attribute__((weak)) HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len);
int32_t __attribute__((weak)) HcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                           uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
int32_t __attribute__((weak)) HcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len);
int32_t __attribute__((weak)) HcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
int32_t __attribute__((weak)) HcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len);
int32_t __attribute__((weak)) HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx);
int32_t __attribute__((weak)) HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx);
int32_t __attribute__((weak)) HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout);
int32_t __attribute__((weak)) HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout);
int32_t __attribute__((weak)) HcommBatchModeStart(const char* batchTag);
int32_t __attribute__((weak)) HcommBatchModeEnd(const char* batchTag);
int32_t __attribute__((weak)) HcommAcquireComm(const char* commId);
int32_t __attribute__((weak)) HcommReleaseComm(const char* commId);
HcclResult __attribute__((weak)) HcommSymWinGetPeerPointer(CommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
int32_t __attribute__((weak)) HcommThreadSynchronize(ThreadHandle thread);
int32_t __attribute__((weak)) HcommSendRequest(MsgHandle handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId);
int32_t __attribute__((weak)) HcommWaitResponse(MsgHandle handle, void* dst, size_t sizeByte, uint32_t* msgId);
int32_t __attribute__((weak)) HcommFlush();
int32_t __attribute__((weak)) HcommChannelFence(ChannelHandle channel);
int32_t __attribute__((weak)) HcommWriteWithNotifyNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
int32_t __attribute__((weak)) HcommFenceOnThread(ThreadHandle thread);
int32_t __attribute__((weak)) HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel);
HcclResult __attribute__((weak)) HcommThreadJoin(ThreadHandle thread, uint32_t timeout);

void HcommPrimitivesDlInit(void* libHcommHandle);  // 本模块独立初始化
void HcommPrimitivesDlFini(void);                  // 本模块独立销毁

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PRIMITIVES_DL_H