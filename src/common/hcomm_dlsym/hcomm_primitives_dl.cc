/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "hcomm_primitives_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
int32_t (*hcommLocalCopyOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommLocalReduceOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = nullptr;
int32_t (*hcommThreadNotifyRecordOnThreadPtr)(ThreadHandle, ThreadHandle, uint32_t) = nullptr;
int32_t (*hcommThreadNotifyWaitOnThreadPtr)(ThreadHandle, uint32_t, uint32_t) = nullptr;
int32_t (*hcommAclrtNotifyRecordOnThreadPtr)(ThreadHandle, uint64_t) = nullptr;
int32_t (*hcommAclrtNotifyWaitOnThreadPtr)(ThreadHandle, uint64_t, uint32_t) = nullptr;
int32_t (*hcommWriteOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommWriteReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = nullptr;
int32_t (*hcommWriteWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t) = nullptr;
int32_t (*hcommWriteReduceWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp, uint32_t) = nullptr;
int32_t (*hcommReadOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommReadReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = nullptr;
int32_t (*hcommWriteNbiPtr)(ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommWriteWithNotifyNbiPtr)(ChannelHandle, void*, const void*, uint64_t, uint32_t) = nullptr;
int32_t (*hcommReadNbiPtr)(ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommChannelNotifyRecordOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t) = nullptr;
int32_t (*hcommChannelNotifyRecordPtr)(ChannelHandle, uint32_t) = nullptr;
int32_t (*hcommChannelNotifyWaitOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t, uint32_t) = nullptr;
int32_t (*hcommChannelNotifyWaitPtr)(ChannelHandle, uint32_t, uint32_t) = nullptr;
int32_t (*hcommBatchModeStartPtr)(const char*) = nullptr;
int32_t (*hcommBatchModeEndPtr)(const char*) = nullptr;
int32_t (*hcommAcquireCommPtr)(const char*) = nullptr;
int32_t (*hcommReleaseCommPtr)(const char*) = nullptr;
HcclResult (*hcommSymWinGetPeerPointerPtr)(CommSymWindow, size_t, uint32_t, void**) = nullptr;
int32_t (*hcommThreadSynchronizePtr)(ThreadHandle) = nullptr;
int32_t (*hcommSendRequestPtr)(MsgHandle, const char*, const void*, size_t, uint32_t*) = nullptr;
int32_t (*hcommWaitResponsePtr)(MsgHandle, void*, size_t, uint32_t*) = nullptr;
int32_t (*hcommFlushPtr)() = nullptr;
int32_t (*hcommChannelFencePtr)(ChannelHandle) = nullptr;
int32_t (*hcommWriteWithNotifyNbiOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t) = nullptr;
int32_t (*hcommFenceOnThreadPtr)(ThreadHandle) = nullptr;
int32_t (*hcommChannelFenceOnThreadPtr)(ThreadHandle, ChannelHandle) = nullptr;
HcclResult (*hcommThreadJoinPtr)(ThreadHandle, uint32_t) = nullptr;

// 添加支持标志（静态，默认 false）
static bool g_hcommLocalCopyOnThreadSupported = false;
static bool g_hcommLocalReduceOnThreadSupported = false;
static bool g_hcommThreadNotifyRecordOnThreadSupported = false;
static bool g_hcommThreadNotifyWaitOnThreadSupported = false;
static bool g_hcommAclrtNotifyRecordOnThreadSupported = false;
static bool g_hcommAclrtNotifyWaitOnThreadSupported = false;
static bool g_hcommWriteOnThreadSupported = false;
static bool g_hcommWriteReduceOnThreadSupported = false;
static bool g_hcommWriteWithNotifyOnThreadSupported = false;
static bool g_hcommWriteReduceWithNotifyOnThreadSupported = false;
static bool g_hcommReadOnThreadSupported = false;
static bool g_hcommReadReduceOnThreadSupported = false;
static bool g_hcommWriteNbiSupported = false;
static bool g_hcommWriteWithNotifyNbiSupported = false;
static bool g_hcommReadNbiSupported = false;
static bool g_hcommChannelNotifyRecordOnThreadSupported = false;
static bool g_hcommChannelNotifyRecordSupported = false;
static bool g_hcommChannelNotifyWaitOnThreadSupported = false;
static bool g_hcommChannelNotifyWaitSupported = false;
static bool g_hcommBatchModeStartSupported = false;
static bool g_hcommBatchModeEndSupported = false;
static bool g_hcommAcquireCommSupported = false;
static bool g_hcommReleaseCommSupported = false;
static bool g_hcommSymWinGetPeerPointerSupported = false;
static bool g_hcommThreadSynchronizeSupported = false;
static bool g_hcommSendRequestSupported = false;
static bool g_hcommWaitResponseSupported = false;
static bool g_hcommFlushSupported = false;
static bool g_hcommChannelFenceSupported = false;
static bool g_hcommWriteWithNotifyNbiOnThreadSupported = false;
static bool g_hcommFenceOnThreadSupported = false;
static bool g_hcommChannelFenceOnThreadSupported = false;
static bool g_hcommThreadJoinSupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
int32_t __attribute__((weak)) HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommLocalCopyOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommLocalReduceOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t count,
                                            HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    HCCL_ERROR("[HcclWrapper] HcommLocalReduceOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx) {
    (void)thread; (void)dstThread; (void)dstNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommThreadNotifyRecordOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut) {
    (void)thread; (void)notifyIdx; (void)timeOut;
    HCCL_ERROR("[HcclWrapper] HcommThreadNotifyWaitOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId) {
    (void)thread; (void)dstNotifyId;
    HCCL_ERROR("[HcclWrapper] HcommAclrtNotifyRecordOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut) {
    (void)thread; (void)notifyId; (void)timeOut;
    HCCL_ERROR("[HcclWrapper] HcommAclrtNotifyWaitOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommWriteOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                            uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    HCCL_ERROR("[HcclWrapper] HcommWriteReduceOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                uint64_t len, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteWithNotifyOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                      uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp,
                                                      uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteReduceWithNotifyOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommReadOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                           uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    HCCL_ERROR("[HcclWrapper] HcommReadReduceOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommWriteNbi not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx) {
    (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteWithNotifyNbi not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommReadNbi not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyRecordOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx) {
    (void)channel; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyRecord not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) {
    (void)thread; (void)channel; (void)localNotifyIdx; (void)timeout;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyWaitOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) {
    (void)channel; (void)localNotifyIdx; (void)timeout;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyWait not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommBatchModeStart(const char* batchTag) {
    (void)batchTag;
    HCCL_ERROR("[HcclWrapper] HcommBatchModeStart not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommBatchModeEnd(const char* batchTag) {
    (void)batchTag;
    HCCL_ERROR("[HcclWrapper] HcommBatchModeEnd not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommAcquireComm(const char* commId) {
    (void)commId;
    HCCL_ERROR("[HcclWrapper] HcommAcquireComm not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommReleaseComm(const char* commId) {
    (void)commId;
    HCCL_ERROR("[HcclWrapper] HcommReleaseComm not supported");
    return -1;
}

HcclResult __attribute__((weak)) HcommSymWinGetPeerPointer(CommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr) {
    (void)winHandle; (void)offset; (void)peerRank; (void)ptr;
    HCCL_ERROR("[HcclWrapper] HcommSymWinGetPeerPointer not supported");
    return HCCL_E_NOT_SUPPORT;
}

int32_t __attribute__((weak)) HcommThreadSynchronize(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommThreadSynchronize not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommSendRequest(MsgHandle handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId) {
    (void)handle; (void)msgTag; (void)src; (void)sizeByte; (void)msgId;
    HCCL_ERROR("[HcclWrapper] HcommSendRequest not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWaitResponse(MsgHandle handle, void* dst, size_t sizeByte, uint32_t* msgId) {
    (void)handle; (void)dst; (void)sizeByte; (void)msgId;
    HCCL_ERROR("[HcclWrapper] HcommWaitResponse not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommFlush() {
    HCCL_ERROR("[HcclWrapper] HcommFlush not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommChannelFence(ChannelHandle channel) {
    (void)channel;
    HCCL_ERROR("[HcclWrapper] HcommChannelFence not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommWriteWithNotifyNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteWithNotifyNbiOnThread not supported");
    return -1;
}

int32_t __attribute__((weak)) HcommFenceOnThread(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommFenceOnThread not supported");
    return -1;
}
int32_t __attribute__((weak)) HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel) {
    (void)channel;
    HCCL_ERROR("[HcclWrapper] HcommChannelFenceOnThread not supported");
    return -1;
}

HcclResult __attribute__((weak)) HcommThreadJoin(ThreadHandle thread, uint32_t timeout) {
    (void)thread; (void)timeout;
    HCCL_ERROR("[HcclWrapper] HcommThreadJoin not supported");
    return HCCL_E_NOT_SUPPORT;
}

// ---------- 初始化函数 ----------
void HcommPrimitivesDlInit(void* libHcommHandle) {
    // 辅助宏：解析符号，失败则指向对应桩函数，同时设置支持标志
    #define SET_PTR(ptr, handle, name, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(handle, name); \
            if (ptr == nullptr) { \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcommLocalCopyOnThreadPtr, libHcommHandle, "HcommLocalCopyOnThread", g_hcommLocalCopyOnThreadSupported);
    SET_PTR(hcommLocalReduceOnThreadPtr, libHcommHandle, "HcommLocalReduceOnThread", g_hcommLocalReduceOnThreadSupported);
    SET_PTR(hcommThreadNotifyRecordOnThreadPtr, libHcommHandle, "HcommThreadNotifyRecordOnThread", g_hcommThreadNotifyRecordOnThreadSupported);
    SET_PTR(hcommThreadNotifyWaitOnThreadPtr, libHcommHandle, "HcommThreadNotifyWaitOnThread", g_hcommThreadNotifyWaitOnThreadSupported);
    SET_PTR(hcommAclrtNotifyRecordOnThreadPtr, libHcommHandle, "HcommAclrtNotifyRecordOnThread", g_hcommAclrtNotifyRecordOnThreadSupported);
    SET_PTR(hcommAclrtNotifyWaitOnThreadPtr, libHcommHandle, "HcommAclrtNotifyWaitOnThread", g_hcommAclrtNotifyWaitOnThreadSupported);
    SET_PTR(hcommWriteOnThreadPtr, libHcommHandle, "HcommWriteOnThread", g_hcommWriteOnThreadSupported);
    SET_PTR(hcommWriteReduceOnThreadPtr, libHcommHandle, "HcommWriteReduceOnThread", g_hcommWriteReduceOnThreadSupported);
    SET_PTR(hcommWriteWithNotifyOnThreadPtr, libHcommHandle, "HcommWriteWithNotifyOnThread", g_hcommWriteWithNotifyOnThreadSupported);
    SET_PTR(hcommWriteReduceWithNotifyOnThreadPtr, libHcommHandle, "HcommWriteReduceWithNotifyOnThread", g_hcommWriteReduceWithNotifyOnThreadSupported);
    SET_PTR(hcommReadOnThreadPtr, libHcommHandle, "HcommReadOnThread", g_hcommReadOnThreadSupported);
    SET_PTR(hcommReadReduceOnThreadPtr, libHcommHandle, "HcommReadReduceOnThread", g_hcommReadReduceOnThreadSupported);
    SET_PTR(hcommWriteNbiPtr, libHcommHandle, "HcommWriteNbi", g_hcommWriteNbiSupported);
    SET_PTR(hcommWriteWithNotifyNbiPtr, libHcommHandle, "HcommWriteWithNotifyNbi", g_hcommWriteWithNotifyNbiSupported);
    SET_PTR(hcommReadNbiPtr, libHcommHandle, "HcommReadNbi", g_hcommReadNbiSupported);
    SET_PTR(hcommChannelNotifyRecordOnThreadPtr, libHcommHandle, "HcommChannelNotifyRecordOnThread", g_hcommChannelNotifyRecordOnThreadSupported);
    SET_PTR(hcommChannelNotifyRecordPtr, libHcommHandle, "HcommChannelNotifyRecord", g_hcommChannelNotifyRecordSupported);
    SET_PTR(hcommChannelNotifyWaitOnThreadPtr, libHcommHandle, "HcommChannelNotifyWaitOnThread", g_hcommChannelNotifyWaitOnThreadSupported);
    SET_PTR(hcommChannelNotifyWaitPtr, libHcommHandle, "HcommChannelNotifyWait", g_hcommChannelNotifyWaitSupported);
    SET_PTR(hcommBatchModeStartPtr, libHcommHandle, "HcommBatchModeStart", g_hcommBatchModeStartSupported);
    SET_PTR(hcommBatchModeEndPtr, libHcommHandle, "HcommBatchModeEnd", g_hcommBatchModeEndSupported);
    SET_PTR(hcommAcquireCommPtr, libHcommHandle, "HcommAcquireComm", g_hcommAcquireCommSupported);
    SET_PTR(hcommReleaseCommPtr, libHcommHandle, "HcommReleaseComm", g_hcommReleaseCommSupported);
    SET_PTR(hcommSymWinGetPeerPointerPtr, libHcommHandle, "HcommSymWinGetPeerPointer", g_hcommSymWinGetPeerPointerSupported);
    SET_PTR(hcommThreadSynchronizePtr, libHcommHandle, "HcommThreadSynchronize", g_hcommThreadSynchronizeSupported);
    SET_PTR(hcommSendRequestPtr, libHcommHandle, "HcommSendRequest", g_hcommSendRequestSupported);
    SET_PTR(hcommWaitResponsePtr, libHcommHandle, "HcommWaitResponse", g_hcommWaitResponseSupported);
    SET_PTR(hcommFlushPtr, libHcommHandle, "HcommFlush", g_hcommFlushSupported);
    SET_PTR(hcommChannelFencePtr, libHcommHandle, "HcommChannelFence", g_hcommChannelFenceSupported);
    SET_PTR(hcommWriteWithNotifyNbiOnThreadPtr, libHcommHandle, "HcommWriteWithNotifyNbiOnThread", g_hcommWriteWithNotifyNbiOnThreadSupported);
    SET_PTR(hcommFenceOnThreadPtr, libHcommHandle, "HcommFenceOnThread", g_hcommFenceOnThreadSupported);
    SET_PTR(hcommChannelFenceOnThreadPtr, libHcommHandle, "HcommChannelFenceOnThread", g_hcommChannelFenceOnThreadSupported);
    SET_PTR(hcommThreadJoinPtr, libHcommHandle, "HcommThreadJoin", g_hcommThreadJoinSupported);

    #undef SET_PTR
}

void HcommPrimitivesDlFini(void) {
    // hcommLocalCopyOnThreadPtr = StubHcommLocalCopyOnThread;
    // hcommLocalReduceOnThreadPtr = StubHcommLocalReduceOnThread;
    // hcommThreadNotifyRecordOnThreadPtr = StubHcommThreadNotifyRecordOnThread;
    // hcommThreadNotifyWaitOnThreadPtr = StubHcommThreadNotifyWaitOnThread;
    // hcommAclrtNotifyRecordOnThreadPtr = StubHcommAclrtNotifyRecordOnThread;
    // hcommAclrtNotifyWaitOnThreadPtr = StubHcommAclrtNotifyWaitOnThread;
    // hcommWriteOnThreadPtr = StubHcommWriteOnThread;
    // hcommWriteReduceOnThreadPtr = StubHcommWriteReduceOnThread;
    // hcommWriteWithNotifyOnThreadPtr = StubHcommWriteWithNotifyOnThread;
    // hcommWriteReduceWithNotifyOnThreadPtr = StubHcommWriteReduceWithNotifyOnThread;
    // hcommReadOnThreadPtr = StubHcommReadOnThread;
    // hcommReadReduceOnThreadPtr = StubHcommReadReduceOnThread;
    // hcommWriteNbiPtr = StubHcommWriteNbi;
    // hcommWriteWithNotifyNbiPtr = StubHcommWriteWithNotifyNbi;
    // hcommReadNbiPtr = StubHcommReadNbi;
    // hcommChannelNotifyRecordOnThreadPtr = StubHcommChannelNotifyRecordOnThread;
    // hcommChannelNotifyRecordPtr = StubHcommChannelNotifyRecord;
    // hcommChannelNotifyWaitOnThreadPtr = StubHcommChannelNotifyWaitOnThread;
    // hcommChannelNotifyWaitPtr = StubHcommChannelNotifyWait;
    // hcommBatchModeStartPtr = StubHcommBatchModeStart;
    // hcommBatchModeEndPtr = StubHcommBatchModeEnd;
    // hcommAcquireCommPtr = StubHcommAcquireComm;
    // hcommReleaseCommPtr = StubHcommReleaseComm;
    // hcommSymWinGetPeerPointerPtr = StubHcommSymWinGetPeerPointer;
    // hcommThreadSynchronizePtr = StubHcommThreadSynchronize;
    // hcommSendRequestPtr = StubHcommSendRequest;
    // hcommWaitResponsePtr = StubHcommWaitResponse;
    // hcommFlushPtr = StubHcommFlush;
    // hcommChannelFencePtr = StubHcommChannelFence;
    // hcommWriteWithNotifyNbiOnThreadPtr = StubHcommWriteWithNotifyNbiOnThread;
    // hcommFenceOnThreadPtr = StubHcommFenceOnThread;
    // hcommChannelFenceOnThreadPtr = StubHcommChannelFenceOnThread;
    // hcommThreadJoinPtr = StubHcommThreadJoin;
}

// ---------- 对外提供的查询接口（判断函数是否存在）----------
extern "C" bool HcommIsSupportHcommLocalCopyOnThread(void) {
    return g_hcommLocalCopyOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommLocalReduceOnThread(void) {
    return g_hcommLocalReduceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommThreadNotifyRecordOnThread(void) {
    return g_hcommThreadNotifyRecordOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommThreadNotifyWaitOnThread(void) {
    return g_hcommThreadNotifyWaitOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommAclrtNotifyRecordOnThread(void) {
    return g_hcommAclrtNotifyRecordOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommAclrtNotifyWaitOnThread(void) {
    return g_hcommAclrtNotifyWaitOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteOnThread(void) {
    return g_hcommWriteOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteReduceOnThread(void) {
    return g_hcommWriteReduceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteWithNotifyOnThread(void) {
    return g_hcommWriteWithNotifyOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteReduceWithNotifyOnThread(void) {
    return g_hcommWriteReduceWithNotifyOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommReadOnThread(void) {
    return g_hcommReadOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommReadReduceOnThread(void) {
    return g_hcommReadReduceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteNbi(void) {
    return g_hcommWriteNbiSupported;
}
extern "C" bool HcommIsSupportHcommWriteWithNotifyNbi(void) {
    return g_hcommWriteWithNotifyNbiSupported;
}
extern "C" bool HcommIsSupportHcommReadNbi(void) {
    return g_hcommReadNbiSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyRecordOnThread(void) {
    return g_hcommChannelNotifyRecordOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyRecord(void) {
    return g_hcommChannelNotifyRecordSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyWaitOnThread(void) {
    return g_hcommChannelNotifyWaitOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyWait(void) {
    return g_hcommChannelNotifyWaitSupported;
}
extern "C" bool HcommIsSupportHcommBatchModeStart(void) {
    return g_hcommBatchModeStartSupported;
}
extern "C" bool HcommIsSupportHcommBatchModeEnd(void) {
    return g_hcommBatchModeEndSupported;
}
extern "C" bool HcommIsSupportHcommAcquireComm(void) {
    return g_hcommAcquireCommSupported;
}
extern "C" bool HcommIsSupportHcommReleaseComm(void) {
    return g_hcommReleaseCommSupported;
}
extern "C" bool HcommIsSupportHcommSymWinGetPeerPointer(void) {
    return g_hcommSymWinGetPeerPointerSupported;
}
extern "C" bool HcommIsSupportHcommThreadSynchronize(void) {
    return g_hcommThreadSynchronizeSupported;
}
extern "C" bool HcommIsSupportHcommSendRequest(void) {
    return g_hcommSendRequestSupported;
}
extern "C" bool HcommIsSupportHcommWaitResponse(void) {
    return g_hcommWaitResponseSupported;
}
extern "C" bool HcommIsSupportHcommFlush(void) {
    return g_hcommFlushSupported;
}
extern "C" bool HcommIsSupportHcommChannelFence(void) {
    return g_hcommChannelFenceSupported;
}
extern "C" bool HcommIsSupportHcommWriteWithNotifyNbiOnThread(void) {
    return g_hcommWriteWithNotifyNbiOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommFenceOnThread(void) {
    return g_hcommFenceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelFenceOnThread(void) {
    return g_hcommChannelFenceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommThreadJoin(void) {
    return g_hcommThreadJoinSupported;
}