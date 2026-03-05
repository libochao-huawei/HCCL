#include "hcomm_primitives_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
int32_t (*hcommLocalCopyOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommLocalReduceOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = NULL;
int32_t (*hcommThreadNotifyRecordOnThreadPtr)(ThreadHandle, ThreadHandle, uint32_t) = NULL;
int32_t (*hcommThreadNotifyWaitOnThreadPtr)(ThreadHandle, uint32_t, uint32_t) = NULL;
int32_t (*hcommAclrtNotifyRecordOnThreadPtr)(ThreadHandle, uint64_t) = NULL;
int32_t (*hcommAclrtNotifyWaitOnThreadPtr)(ThreadHandle, uint64_t, uint32_t) = NULL;
int32_t (*hcommWriteOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommWriteReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = NULL;
int32_t (*hcommWriteWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t) = NULL;
int32_t (*hcommWriteReduceWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp, uint32_t) = NULL;
int32_t (*hcommReadOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommReadReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = NULL;
int32_t (*hcommWriteNbiPtr)(ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommWriteWithNotifyNbiPtr)(ChannelHandle, void*, const void*, uint64_t, uint32_t) = NULL;
int32_t (*hcommReadNbiPtr)(ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommChannelNotifyRecordOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t) = NULL;
int32_t (*hcommChannelNotifyRecordPtr)(ChannelHandle, uint32_t) = NULL;
int32_t (*hcommChannelNotifyWaitOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t, uint32_t) = NULL;
int32_t (*hcommChannelNotifyWaitPtr)(ChannelHandle, uint32_t, uint32_t) = NULL;
int32_t (*hcommBatchModeStartPtr)(const char*) = NULL;
int32_t (*hcommBatchModeEndPtr)(const char*) = NULL;
int32_t (*hcommAcquireCommPtr)(const char*) = NULL;
int32_t (*hcommReleaseCommPtr)(const char*) = NULL;
HcclResult (*hcommSymWinGetPeerPointerPtr)(CommSymWindow, size_t, uint32_t, void**) = NULL;
int32_t (*hcommThreadSynchronizePtr)(ThreadHandle) = NULL;
int32_t (*hcommSendRequestPtr)(MsgHandle, const char*, const void*, size_t, uint32_t*) = NULL;
int32_t (*hcommWaitResponsePtr)(MsgHandle, void*, size_t, uint32_t*) = NULL;
int32_t (*hcommFlushPtr)() = NULL;
int32_t (*hcommChannelFencePtr)(ChannelHandle) = NULL;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static int32_t StubHcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)dst; (void)src; (void)len;
    fprintf(stderr, "[HcclWrapper] HcommLocalCopyOnThread not supported\n");
    return -1;
}

static int32_t StubHcommLocalReduceOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t count,
                                            HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    fprintf(stderr, "[HcclWrapper] HcommLocalReduceOnThread not supported\n");
    return -1;
}

static int32_t StubHcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx) {
    (void)thread; (void)dstThread; (void)dstNotifyIdx;
    fprintf(stderr, "[HcclWrapper] HcommThreadNotifyRecordOnThread not supported\n");
    return -1;
}

static int32_t StubHcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut) {
    (void)thread; (void)notifyIdx; (void)timeOut;
    fprintf(stderr, "[HcclWrapper] HcommThreadNotifyWaitOnThread not supported\n");
    return -1;
}

static int32_t StubHcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId) {
    (void)thread; (void)dstNotifyId;
    fprintf(stderr, "[HcclWrapper] HcommAclrtNotifyRecordOnThread not supported\n");
    return -1;
}

static int32_t StubHcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut) {
    (void)thread; (void)notifyId; (void)timeOut;
    fprintf(stderr, "[HcclWrapper] HcommAclrtNotifyWaitOnThread not supported\n");
    return -1;
}

static int32_t StubHcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    fprintf(stderr, "[HcclWrapper] HcommWriteOnThread not supported\n");
    return -1;
}

static int32_t StubHcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                            uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    fprintf(stderr, "[HcclWrapper] HcommWriteReduceOnThread not supported\n");
    return -1;
}

static int32_t StubHcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                uint64_t len, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    fprintf(stderr, "[HcclWrapper] HcommWriteWithNotifyOnThread not supported\n");
    return -1;
}

static int32_t StubHcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                      uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp,
                                                      uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp; (void)remoteNotifyIdx;
    fprintf(stderr, "[HcclWrapper] HcommWriteReduceWithNotifyOnThread not supported\n");
    return -1;
}

static int32_t StubHcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    fprintf(stderr, "[HcclWrapper] HcommReadOnThread not supported\n");
    return -1;
}

static int32_t StubHcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                           uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    fprintf(stderr, "[HcclWrapper] HcommReadReduceOnThread not supported\n");
    return -1;
}

static int32_t StubHcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)channel; (void)dst; (void)src; (void)len;
    fprintf(stderr, "[HcclWrapper] HcommWriteNbi not supported\n");
    return -1;
}

static int32_t StubHcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx) {
    (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    fprintf(stderr, "[HcclWrapper] HcommWriteWithNotifyNbi not supported\n");
    return -1;
}

static int32_t StubHcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)channel; (void)dst; (void)src; (void)len;
    fprintf(stderr, "[HcclWrapper] HcommReadNbi not supported\n");
    return -1;
}

static int32_t StubHcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)remoteNotifyIdx;
    fprintf(stderr, "[HcclWrapper] HcommChannelNotifyRecordOnThread not supported\n");
    return -1;
}

static int32_t StubHcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx) {
    (void)channel; (void)remoteNotifyIdx;
    fprintf(stderr, "[HcclWrapper] HcommChannelNotifyRecord not supported\n");
    return -1;
}

static int32_t StubHcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) {
    (void)thread; (void)channel; (void)localNotifyIdx; (void)timeout;
    fprintf(stderr, "[HcclWrapper] HcommChannelNotifyWaitOnThread not supported\n");
    return -1;
}

static int32_t StubHcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) {
    (void)channel; (void)localNotifyIdx; (void)timeout;
    fprintf(stderr, "[HcclWrapper] HcommChannelNotifyWait not supported\n");
    return -1;
}

static int32_t StubHcommBatchModeStart(const char* batchTag) {
    (void)batchTag;
    fprintf(stderr, "[HcclWrapper] HcommBatchModeStart not supported\n");
    return -1;
}

static int32_t StubHcommBatchModeEnd(const char* batchTag) {
    (void)batchTag;
    fprintf(stderr, "[HcclWrapper] HcommBatchModeEnd not supported\n");
    return -1;
}

static int32_t StubHcommAcquireComm(const char* commId) {
    (void)commId;
    fprintf(stderr, "[HcclWrapper] HcommAcquireComm not supported\n");
    return -1;
}

static int32_t StubHcommReleaseComm(const char* commId) {
    (void)commId;
    fprintf(stderr, "[HcclWrapper] HcommReleaseComm not supported\n");
    return -1;
}

static HcclResult StubHcommSymWinGetPeerPointer(CommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr) {
    (void)winHandle; (void)offset; (void)peerRank; (void)ptr;
    fprintf(stderr, "[HcclWrapper] HcommSymWinGetPeerPointer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static int32_t StubHcommThreadSynchronize(ThreadHandle thread) {
    (void)thread;
    fprintf(stderr, "[HcclWrapper] HcommThreadSynchronize not supported\n");
    return -1;
}

static int32_t StubHcommSendRequest(MsgHandle handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId) {
    (void)handle; (void)msgTag; (void)src; (void)sizeByte; (void)msgId;
    fprintf(stderr, "[HcclWrapper] HcommSendRequest not supported\n");
    return -1;
}

static int32_t StubHcommWaitResponse(MsgHandle handle, void* dst, size_t sizeByte, uint32_t* msgId) {
    (void)handle; (void)dst; (void)sizeByte; (void)msgId;
    fprintf(stderr, "[HcclWrapper] HcommWaitResponse not supported\n");
    return -1;
}

static int32_t StubHcommFlush() {
    fprintf(stderr, "[HcclWrapper] HcommFlush not supported\n");
    return -1;
}

static int32_t StubHcommChannelFence(ChannelHandle channel) {
    (void)channel;
    fprintf(stderr, "[HcclWrapper] HcommChannelFence not supported\n");
    return -1;
}

// ---------- 初始化函数 ----------
void HcommPrimitivesDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, name, stub) \
        do { \
            ptr = (typeof(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) ptr = stub; \
        } while(0)

    SET_PTR(hcommLocalCopyOnThreadPtr, "HcommLocalCopyOnThread", StubHcommLocalCopyOnThread);
    SET_PTR(hcommLocalReduceOnThreadPtr, "HcommLocalReduceOnThread", StubHcommLocalReduceOnThread);
    SET_PTR(hcommThreadNotifyRecordOnThreadPtr, "HcommThreadNotifyRecordOnThread", StubHcommThreadNotifyRecordOnThread);
    SET_PTR(hcommThreadNotifyWaitOnThreadPtr, "HcommThreadNotifyWaitOnThread", StubHcommThreadNotifyWaitOnThread);
    SET_PTR(hcommAclrtNotifyRecordOnThreadPtr, "HcommAclrtNotifyRecordOnThread", StubHcommAclrtNotifyRecordOnThread);
    SET_PTR(hcommAclrtNotifyWaitOnThreadPtr, "HcommAclrtNotifyWaitOnThread", StubHcommAclrtNotifyWaitOnThread);
    SET_PTR(hcommWriteOnThreadPtr, "HcommWriteOnThread", StubHcommWriteOnThread);
    SET_PTR(hcommWriteReduceOnThreadPtr, "HcommWriteReduceOnThread", StubHcommWriteReduceOnThread);
    SET_PTR(hcommWriteWithNotifyOnThreadPtr, "HcommWriteWithNotifyOnThread", StubHcommWriteWithNotifyOnThread);
    SET_PTR(hcommWriteReduceWithNotifyOnThreadPtr, "HcommWriteReduceWithNotifyOnThread", StubHcommWriteReduceWithNotifyOnThread);
    SET_PTR(hcommReadOnThreadPtr, "HcommReadOnThread", StubHcommReadOnThread);
    SET_PTR(hcommReadReduceOnThreadPtr, "HcommReadReduceOnThread", StubHcommReadReduceOnThread);
    SET_PTR(hcommWriteNbiPtr, "HcommWriteNbi", StubHcommWriteNbi);
    SET_PTR(hcommWriteWithNotifyNbiPtr, "HcommWriteWithNotifyNbi", StubHcommWriteWithNotifyNbi);
    SET_PTR(hcommReadNbiPtr, "HcommReadNbi", StubHcommReadNbi);
    SET_PTR(hcommChannelNotifyRecordOnThreadPtr, "HcommChannelNotifyRecordOnThread", StubHcommChannelNotifyRecordOnThread);
    SET_PTR(hcommChannelNotifyRecordPtr, "HcommChannelNotifyRecord", StubHcommChannelNotifyRecord);
    SET_PTR(hcommChannelNotifyWaitOnThreadPtr, "HcommChannelNotifyWaitOnThread", StubHcommChannelNotifyWaitOnThread);
    SET_PTR(hcommChannelNotifyWaitPtr, "HcommChannelNotifyWait", StubHcommChannelNotifyWait);
    SET_PTR(hcommBatchModeStartPtr, "HcommBatchModeStart", StubHcommBatchModeStart);
    SET_PTR(hcommBatchModeEndPtr, "HcommBatchModeEnd", StubHcommBatchModeEnd);
    SET_PTR(hcommAcquireCommPtr, "HcommAcquireComm", StubHcommAcquireComm);
    SET_PTR(hcommReleaseCommPtr, "HcommReleaseComm", StubHcommReleaseComm);
    SET_PTR(hcommSymWinGetPeerPointerPtr, "HcommSymWinGetPeerPointer", StubHcommSymWinGetPeerPointer);
    SET_PTR(hcommThreadSynchronizePtr, "HcommThreadSynchronize", StubHcommThreadSynchronize);
    SET_PTR(hcommSendRequestPtr, "HcommSendRequest", StubHcommSendRequest);
    SET_PTR(hcommWaitResponsePtr, "HcommWaitResponse", StubHcommWaitResponse);
    SET_PTR(hcommFlushPtr, "HcommFlush", StubHcommFlush);
    SET_PTR(hcommChannelFencePtr, "HcommChannelFence", StubHcommChannelFence);

    #undef SET_PTR

    if (dlerror()) {
        fprintf(stderr, "[HcclWrapper] Warning: dlerror after symbol resolution\n");
    }
    return 0;
}

void HcommPrimitivesDlFini(void) {
    hcommLocalCopyOnThreadPtr = StubHcommLocalCopyOnThread;
    hcommLocalReduceOnThreadPtr = StubHcommLocalReduceOnThread;
    hcommThreadNotifyRecordOnThreadPtr = StubHcommThreadNotifyRecordOnThread;
    hcommThreadNotifyWaitOnThreadPtr = StubHcommThreadNotifyWaitOnThread;
    hcommAclrtNotifyRecordOnThreadPtr = StubHcommAclrtNotifyRecordOnThread;
    hcommAclrtNotifyWaitOnThreadPtr = StubHcommAclrtNotifyWaitOnThread;
    hcommWriteOnThreadPtr = StubHcommWriteOnThread;
    hcommWriteReduceOnThreadPtr = StubHcommWriteReduceOnThread;
    hcommWriteWithNotifyOnThreadPtr = StubHcommWriteWithNotifyOnThread;
    hcommWriteReduceWithNotifyOnThreadPtr = StubHcommWriteReduceWithNotifyOnThread;
    hcommReadOnThreadPtr = StubHcommReadOnThread;
    hcommReadReduceOnThreadPtr = StubHcommReadReduceOnThread;
    hcommWriteNbiPtr = StubHcommWriteNbi;
    hcommWriteWithNotifyNbiPtr = StubHcommWriteWithNotifyNbi;
    hcommReadNbiPtr = StubHcommReadNbi;
    hcommChannelNotifyRecordOnThreadPtr = StubHcommChannelNotifyRecordOnThread;
    hcommChannelNotifyRecordPtr = StubHcommChannelNotifyRecord;
    hcommChannelNotifyWaitOnThreadPtr = StubHcommChannelNotifyWaitOnThread;
    hcommChannelNotifyWaitPtr = StubHcommChannelNotifyWait;
    hcommBatchModeStartPtr = StubHcommBatchModeStart;
    hcommBatchModeEndPtr = StubHcommBatchModeEnd;
    hcommAcquireCommPtr = StubHcommAcquireComm;
    hcommReleaseCommPtr = StubHcommReleaseComm;
    hcommSymWinGetPeerPointerPtr = StubHcommSymWinGetPeerPointer;
    hcommThreadSynchronizePtr = StubHcommThreadSynchronize;
    hcommSendRequestPtr = StubHcommSendRequest;
    hcommWaitResponsePtr = StubHcommWaitResponse;
    hcommFlushPtr = StubHcommFlush;
    hcommChannelFencePtr = StubHcommChannelFence;
}