#include "log.h"
#include "hccl_mc2_ex_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（静态）
static HcclResult (*hcclGetCommHandleByCtxPtr)(void*, void**) = NULL;
static HcclResult (*hcclReleaseCommPtr)(void*) = NULL;
static HcclResult (*hcclGetTaskStatusPtr)(void*, void*) = NULL;
static HcclResult (*hcclCheckFinishByStreamPtr)(void*) = NULL;
static HcclResult (*hcclPrintTaskExceptionAllCommPtr)(void*) = NULL;
static HcclResult (*hcclLaunchCcoreWaitPtr)(void*, uint64_t, uint32_t, uint64_t, bool) = NULL;
static HcclResult (*hcclLaunchCcorePostPtr)(void*, uint64_t, uint32_t, uint64_t) = NULL;
static HcclResult (*hcclLaunchOpPtr)(void*, void*) = NULL;

// 支持标志（静态，默认 false）
static bool g_hcclGetCommHandleByCtxSupported = false;
static bool g_hcclReleaseCommSupported = false;
static bool g_hcclGetTaskStatusSupported = false;
static bool g_hcclCheckFinishByStreamSupported = false;
static bool g_hcclPrintTaskExceptionAllCommSupported = false;
static bool g_hcclLaunchCcoreWaitSupported = false;
static bool g_hcclLaunchCcorePostSupported = false;
static bool g_hcclLaunchOpSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcclGetCommHandleByCtx(void* ctx, void** opHandle) {
    (void)ctx; (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclGetCommHandleByCtx not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclReleaseComm(void* opHandle) {
    (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclReleaseComm not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetTaskStatus(void* opHandle, void* status) {
    (void)opHandle; (void)status;
    HCCL_ERROR("[HcclWrapper] HcclGetTaskStatus not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCheckFinishByStream(void* opHandle) {
    (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclCheckFinishByStream not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclPrintTaskExceptionAllComm(void* opHandle) {
    (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclPrintTaskExceptionAllComm not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast) {
    (void)opHandle; (void)waitAddr; (void)turnNum; (void)turnNumAddr; (void)isLast;
    HCCL_ERROR("[HcclWrapper] HcclLaunchCcoreWait not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr) {
    (void)opHandle; (void)recordAddr; (void)turnNum; (void)turnNumAddr;
    HCCL_ERROR("[HcclWrapper] HcclLaunchCcorePost not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclLaunchOp(void* opHandle, void* data) {
    (void)opHandle; (void)data;
    HCCL_ERROR("[HcclWrapper] HcclLaunchOp not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// ---------- 初始化函数 ----------
void HcclMc2ExDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcclGetCommHandleByCtxPtr, "HcclGetCommHandleByCtx", StubHcclGetCommHandleByCtx, g_hcclGetCommHandleByCtxSupported);
    SET_PTR(hcclReleaseCommPtr, "HcclReleaseComm", StubHcclReleaseComm, g_hcclReleaseCommSupported);
    SET_PTR(hcclGetTaskStatusPtr, "HcclGetTaskStatus", StubHcclGetTaskStatus, g_hcclGetTaskStatusSupported);
    SET_PTR(hcclCheckFinishByStreamPtr, "HcclCheckFinishByStream", StubHcclCheckFinishByStream, g_hcclCheckFinishByStreamSupported);
    SET_PTR(hcclPrintTaskExceptionAllCommPtr, "HcclPrintTaskExceptionAllComm", StubHcclPrintTaskExceptionAllComm, g_hcclPrintTaskExceptionAllCommSupported);
    SET_PTR(hcclLaunchCcoreWaitPtr, "HcclLaunchCcoreWait", StubHcclLaunchCcoreWait, g_hcclLaunchCcoreWaitSupported);
    SET_PTR(hcclLaunchCcorePostPtr, "HcclLaunchCcorePost", StubHcclLaunchCcorePost, g_hcclLaunchCcorePostSupported);
    SET_PTR(hcclLaunchOpPtr, "HcclLaunchOp", StubHcclLaunchOp, g_hcclLaunchOpSupported);

    #undef SET_PTR
}

void HcclMc2ExDlFini(void) {
    #define RESET_PTR(ptr, stub, support_flag) do { ptr = stub; support_flag = false; } while(0)

    RESET_PTR(hcclGetCommHandleByCtxPtr, StubHcclGetCommHandleByCtx, g_hcclGetCommHandleByCtxSupported);
    RESET_PTR(hcclReleaseCommPtr, StubHcclReleaseComm, g_hcclReleaseCommSupported);
    RESET_PTR(hcclGetTaskStatusPtr, StubHcclGetTaskStatus, g_hcclGetTaskStatusSupported);
    RESET_PTR(hcclCheckFinishByStreamPtr, StubHcclCheckFinishByStream, g_hcclCheckFinishByStreamSupported);
    RESET_PTR(hcclPrintTaskExceptionAllCommPtr, StubHcclPrintTaskExceptionAllComm, g_hcclPrintTaskExceptionAllCommSupported);
    RESET_PTR(hcclLaunchCcoreWaitPtr, StubHcclLaunchCcoreWait, g_hcclLaunchCcoreWaitSupported);
    RESET_PTR(hcclLaunchCcorePostPtr, StubHcclLaunchCcorePost, g_hcclLaunchCcorePostSupported);
    RESET_PTR(hcclLaunchOpPtr, StubHcclLaunchOp, g_hcclLaunchOpSupported);

    #undef RESET_PTR
}

// ---------- 对外API实现 ----------
HcclResult HcclGetCommHandleByCtx(void* ctx, void** opHandle) {
    return hcclGetCommHandleByCtxPtr(ctx, opHandle);
}
HcclResult HcclReleaseComm(void* opHandle) {
    return hcclReleaseCommPtr(opHandle);
}
HcclResult HcclGetTaskStatus(void* opHandle, void* status) {
    return hcclGetTaskStatusPtr(opHandle, status);
}
HcclResult HcclCheckFinishByStream(void* opHandle) {
    return hcclCheckFinishByStreamPtr(opHandle);
}
HcclResult HcclPrintTaskExceptionAllComm(void* opHandle) {
    return hcclPrintTaskExceptionAllCommPtr(opHandle);
}
HcclResult HcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast) {
    return hcclLaunchCcoreWaitPtr(opHandle, waitAddr, turnNum, turnNumAddr, isLast);
}
HcclResult HcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr) {
    return hcclLaunchCcorePostPtr(opHandle, recordAddr, turnNum, turnNumAddr);
}
HcclResult HcclLaunchOp(void* opHandle, void* data) {
    return hcclLaunchOpPtr(opHandle, data);
}

// ---------- 查询函数实现 ----------
extern "C" bool HcommIsSupportHcclGetCommHandleByCtx(void) {
    return g_hcclGetCommHandleByCtxSupported;
}
extern "C" bool HcommIsSupportHcclReleaseComm(void) {
    return g_hcclReleaseCommSupported;
}
extern "C" bool HcommIsSupportHcclGetTaskStatus(void) {
    return g_hcclGetTaskStatusSupported;
}
extern "C" bool HcommIsSupportHcclCheckFinishByStream(void) {
    return g_hcclCheckFinishByStreamSupported;
}
extern "C" bool HcommIsSupportHcclPrintTaskExceptionAllComm(void) {
    return g_hcclPrintTaskExceptionAllCommSupported;
}
extern "C" bool HcommIsSupportHcclLaunchCcoreWait(void) {
    return g_hcclLaunchCcoreWaitSupported;
}
extern "C" bool HcommIsSupportHcclLaunchCcorePost(void) {
    return g_hcclLaunchCcorePostSupported;
}
extern "C" bool HcommIsSupportHcclLaunchOp(void) {
    return g_hcclLaunchOpSupported;
}