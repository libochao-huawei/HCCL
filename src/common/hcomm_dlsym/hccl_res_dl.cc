#include "hccl_res_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
HcclResult (*hcclGetHcclBufferPtr)(HcclComm, void**, uint64_t*) = NULL;
HcclResult (*hcclGetRemoteIpcHcclBufPtr)(HcclComm, uint64_t, void**, uint64_t*) = NULL;
HcclResult (*hcclThreadAcquirePtr)(HcclComm, CommEngine, uint32_t, uint32_t, ThreadHandle*) = NULL;
HcclResult (*hcclThreadAcquireWithStreamPtr)(HcclComm, CommEngine, aclrtStream, uint32_t, ThreadHandle*) = NULL;
HcclResult (*hcclChannelAcquirePtr)(HcclComm, CommEngine, const HcclChannelDesc*, uint32_t, ChannelHandle*) = NULL;
HcclResult (*hcclChannelGetHcclBufferPtr)(HcclComm, ChannelHandle, void**, uint64_t*) = NULL;
HcclResult (*hcclEngineCtxCreatePtr)(HcclComm, const char*, CommEngine, uint64_t, void**) = NULL;
HcclResult (*hcclEngineCtxGetPtr)(HcclComm, const char*, CommEngine, void**, uint64_t*) = NULL;
HcclResult (*hcclEngineCtxCopyPtr)(HcclComm, CommEngine, const char*, const void*, uint64_t, uint64_t) = NULL;
int32_t    (*hcclTaskRegisterPtr)(HcclComm, const char*, Callback) = NULL;
int32_t    (*hcclTaskUnRegisterPtr)(HcclComm, const char*) = NULL;
HcclResult (*hcclDevMemAcquirePtr)(HcclComm, const char*, uint64_t*, void**, bool*) = NULL;
HcclResult (*hcclThreadExportToCommEnginePtr)(HcclComm, uint32_t, const ThreadHandle*, CommEngine, ThreadHandle*) = NULL;
HcclResult (*hcclChannelGetRemoteMemsPtr)(HcclComm, ChannelHandle, uint32_t*, CommMem**, char***) = NULL;
HcclResult (*hcclCommMemRegPtr)(HcclComm, const char*, const CommMem*, HcclMemHandle*) = NULL;
HcclResult (*hcclEngineCtxDestroyPtr)(HcclComm, const char*, CommEngine) = NULL;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size) {
    (void)comm; (void)buffer; (void)size;
    fprintf(stderr, "[HcclWrapper] HcclGetHcclBuffer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclGetRemoteIpcHcclBuf(HcclComm comm, uint64_t remoteRank, void** addr, uint64_t* size) {
    (void)comm; (void)remoteRank; (void)addr; (void)size;
    fprintf(stderr, "[HcclWrapper] HcclGetRemoteIpcHcclBuf not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
                                        uint32_t notifyNumPerThread, ThreadHandle* threads) {
    (void)comm; (void)engine; (void)threadNum; (void)notifyNumPerThread; (void)threads;
    fprintf(stderr, "[HcclWrapper] HcclThreadAcquire not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream,
                                                  uint32_t notifyNum, ThreadHandle* thread) {
    (void)comm; (void)engine; (void)stream; (void)notifyNum; (void)thread;
    fprintf(stderr, "[HcclWrapper] HcclThreadAcquireWithStream not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs,
                                         uint32_t channelNum, ChannelHandle* channels) {
    (void)comm; (void)engine; (void)channelDescs; (void)channelNum; (void)channels;
    fprintf(stderr, "[HcclWrapper] HcclChannelAcquire not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void** buffer, uint64_t* size) {
    (void)comm; (void)channel; (void)buffer; (void)size;
    fprintf(stderr, "[HcclWrapper] HcclChannelGetHcclBuffer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine,
                                          uint64_t size, void** ctx) {
    (void)comm; (void)ctxTag; (void)engine; (void)size; (void)ctx;
    fprintf(stderr, "[HcclWrapper] HcclEngineCtxCreate not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine,
                                       void** ctx, uint64_t* size) {
    (void)comm; (void)ctxTag; (void)engine; (void)ctx; (void)size;
    fprintf(stderr, "[HcclWrapper] HcclEngineCtxGet not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char* ctxTag,
                                        const void* srcCtx, uint64_t size, uint64_t dstCtxOffset) {
    (void)comm; (void)engine; (void)ctxTag; (void)srcCtx; (void)size; (void)dstCtxOffset;
    fprintf(stderr, "[HcclWrapper] HcclEngineCtxCopy not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static int32_t StubHcclTaskRegister(HcclComm comm, const char* msgTag, Callback cb) {
    (void)comm; (void)msgTag; (void)cb;
    fprintf(stderr, "[HcclWrapper] HcclTaskRegister not supported\n");
    return -1;
}

static int32_t StubHcclTaskUnRegister(HcclComm comm, const char* msgTag) {
    (void)comm; (void)msgTag;
    fprintf(stderr, "[HcclWrapper] HcclTaskUnRegister not supported\n");
    return -1;
}

static HcclResult StubHcclDevMemAcquire(HcclComm comm, const char* memTag, uint64_t* size,
                                        void** addr, bool* newCreated) {
    (void)comm; (void)memTag; (void)size; (void)addr; (void)newCreated;
    fprintf(stderr, "[HcclWrapper] HcclDevMemAcquire not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclThreadExportToCommEngine(HcclComm comm, uint32_t threadNum,
                                                   const ThreadHandle* threads, CommEngine dstCommEngine,
                                                   ThreadHandle* exportedThreads) {
    (void)comm; (void)threadNum; (void)threads; (void)dstCommEngine; (void)exportedThreads;
    fprintf(stderr, "[HcclWrapper] HcclThreadExportToCommEngine not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel,
                                               uint32_t* memNum, CommMem** remoteMems, char*** memTags) {
    (void)comm; (void)channel; (void)memNum; (void)remoteMems; (void)memTags;
    fprintf(stderr, "[HcclWrapper] HcclChannelGetRemoteMems not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem,
                                     HcclMemHandle* memHandle) {
    (void)comm; (void)memTag; (void)mem; (void)memHandle;
    fprintf(stderr, "[HcclWrapper] HcclCommMemReg not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine) {
    (void)comm; (void)ctxTag; (void)engine;
    fprintf(stderr, "[HcclWrapper] HcclEngineCtxDestroy not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcclResDlInit(void* libHcommHandle) {
    // 辅助宏：解析符号，失败则指向对应桩函数
    #define SET_PTR(ptr, name, stub) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) ptr = stub; \
        } while(0)

    SET_PTR(hcclGetHcclBufferPtr, "HcclGetHcclBuffer", StubHcclGetHcclBuffer);
    SET_PTR(hcclGetRemoteIpcHcclBufPtr, "HcclGetRemoteIpcHcclBuf", StubHcclGetRemoteIpcHcclBuf);
    SET_PTR(hcclThreadAcquirePtr, "HcclThreadAcquire", StubHcclThreadAcquire);
    SET_PTR(hcclThreadAcquireWithStreamPtr, "HcclThreadAcquireWithStream", StubHcclThreadAcquireWithStream);
    SET_PTR(hcclChannelAcquirePtr, "HcclChannelAcquire", StubHcclChannelAcquire);
    SET_PTR(hcclChannelGetHcclBufferPtr, "HcclChannelGetHcclBuffer", StubHcclChannelGetHcclBuffer);
    SET_PTR(hcclEngineCtxCreatePtr, "HcclEngineCtxCreate", StubHcclEngineCtxCreate);
    SET_PTR(hcclEngineCtxGetPtr, "HcclEngineCtxGet", StubHcclEngineCtxGet);
    SET_PTR(hcclEngineCtxCopyPtr, "HcclEngineCtxCopy", StubHcclEngineCtxCopy);
    SET_PTR(hcclTaskRegisterPtr, "HcclTaskRegister", StubHcclTaskRegister);
    SET_PTR(hcclTaskUnRegisterPtr, "HcclTaskUnRegister", StubHcclTaskUnRegister);
    SET_PTR(hcclDevMemAcquirePtr, "HcclDevMemAcquire", StubHcclDevMemAcquire);
    SET_PTR(hcclThreadExportToCommEnginePtr, "HcclThreadExportToCommEngine", StubHcclThreadExportToCommEngine);
    SET_PTR(hcclChannelGetRemoteMemsPtr, "HcclChannelGetRemoteMems", StubHcclChannelGetRemoteMems);
    SET_PTR(hcclCommMemRegPtr, "HcclCommMemReg", StubHcclCommMemReg);
    SET_PTR(hcclEngineCtxDestroyPtr, "HcclEngineCtxDestroy", StubHcclEngineCtxDestroy);

    #undef SET_PTR

    if (dlerror()) {
        fprintf(stderr, "[HcclWrapper] Warning: dlerror after symbol resolution\n");
    }
}

void HcclResDlFini(void) {
    // 重置为桩函数，防止fini后误用
    hcclGetHcclBufferPtr = StubHcclGetHcclBuffer;
    hcclGetRemoteIpcHcclBufPtr = StubHcclGetRemoteIpcHcclBuf;
    hcclThreadAcquirePtr = StubHcclThreadAcquire;
    hcclThreadAcquireWithStreamPtr = StubHcclThreadAcquireWithStream;
    hcclChannelAcquirePtr = StubHcclChannelAcquire;
    hcclChannelGetHcclBufferPtr = StubHcclChannelGetHcclBuffer;
    hcclEngineCtxCreatePtr = StubHcclEngineCtxCreate;
    hcclEngineCtxGetPtr = StubHcclEngineCtxGet;
    hcclEngineCtxCopyPtr = StubHcclEngineCtxCopy;
    hcclTaskRegisterPtr = StubHcclTaskRegister;
    hcclTaskUnRegisterPtr = StubHcclTaskUnRegister;
    hcclDevMemAcquirePtr = StubHcclDevMemAcquire;
    hcclThreadExportToCommEnginePtr = StubHcclThreadExportToCommEngine;
    hcclChannelGetRemoteMemsPtr = StubHcclChannelGetRemoteMems;
    hcclCommMemRegPtr = StubHcclCommMemReg;
    hcclEngineCtxDestroyPtr = StubHcclEngineCtxDestroy;
}