#include "log.h"
#include "hccl_ex_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcclCreateComResourcePtr)(const char*, u32, void**) = NULL;
HcclResult (*hcclGetAicpuOpStreamNotifyPtr)(const char*, rtStream_t*, void**) = NULL;
HcclResult (*hcclAllocComResourcePtr)(HcclComm, u32, void**) = NULL;
HcclResult (*hcclAllocComResourceByTilingPtr)(HcclComm, void*, void*, void**) = NULL;
HcclResult (*hcclGetAicpuOpStreamAndNotifyPtr)(HcclComm, rtStream_t*, u8, void**) = NULL;
HcclResult (*hcclGetTopoDescPtr)(HcclComm, HcclTopoDescs*, uint32_t) = NULL;
HcclResult (*hcclCommRegisterPtr)(HcclComm, void*, uint64_t, void**, uint32_t) = NULL;
HcclResult (*hcclCommDeregisterPtr)(HcclComm, void*) = NULL;
HcclResult (*hcclCommExchangeMemPtr)(HcclComm, void*, uint32_t*, uint32_t) = NULL;

// 添加支持标志
static bool g_hcclCreateComResourceSupported = false;
static bool g_hcclGetAicpuOpStreamNotifySupported = false;
static bool g_hcclAllocComResourceSupported = false;
static bool g_hcclAllocComResourceByTilingSupported = false;
static bool g_hcclGetAicpuOpStreamAndNotifySupported = false;
static bool g_hcclGetTopoDescSupported = false;
static bool g_hcclCommRegisterSupported = false;
static bool g_hcclCommDeregisterSupported = false;
static bool g_hcclCommExchangeMemSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcclCreateComResource(const char* commName, u32 streamMode, void** commContext) {
    (void)commName; (void)streamMode; (void)commContext;
    HCCL_ERROR("[HcclWrapper] HcclCreateComResource not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetAicpuOpStreamNotify(const char* commName, rtStream_t* Opstream, void** aicpuNotify) {
    (void)commName; (void)Opstream; (void)aicpuNotify;
    HCCL_ERROR("[HcclWrapper] HcclGetAicpuOpStreamNotify not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclAllocComResource(HcclComm comm, u32 streamMode, void** commContext) {
    (void)comm; (void)streamMode; (void)commContext;
    HCCL_ERROR("[HcclWrapper] HcclAllocComResource not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclAllocComResourceByTiling(HcclComm comm, void* stream, void* Mc2Tiling, void** commContext) {
    (void)comm; (void)stream; (void)Mc2Tiling; (void)commContext;
    HCCL_ERROR("[HcclWrapper] HcclAllocComResourceByTiling not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetAicpuOpStreamAndNotify(HcclComm comm, rtStream_t* opstream, u8 aicpuNotifyNum, void** aicpuNotify) {
    (void)comm; (void)opstream; (void)aicpuNotifyNum; (void)aicpuNotify;
    HCCL_ERROR("[HcclWrapper] HcclGetAicpuOpStreamAndNotify not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetTopoDesc(HcclComm comm, HcclTopoDescs* topoDescs, uint32_t topoSize) {
    (void)comm; (void)topoDescs; (void)topoSize;
    HCCL_ERROR("[HcclWrapper] HcclGetTopoDesc not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommRegister(HcclComm comm, void* addr, uint64_t size, void** handle, uint32_t flag) {
    (void)comm; (void)addr; (void)size; (void)handle; (void)flag;
    HCCL_ERROR("[HcclWrapper] HcclCommRegister not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommDeregister(HcclComm comm, void* handle) {
    (void)comm; (void)handle;
    HCCL_ERROR("[HcclWrapper] HcclCommDeregister not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommExchangeMem(HcclComm comm, void* windowHandle, uint32_t* peerRanks, uint32_t peerRankNum) {
    (void)comm; (void)windowHandle; (void)peerRanks; (void)peerRankNum;
    HCCL_ERROR("[HcclWrapper] HcclCommExchangeMem not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcclExDlInit(void* libHcommHandle) {
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

    SET_PTR(hcclCreateComResourcePtr, "HcclCreateComResource", StubHcclCreateComResource, g_hcclCreateComResourceSupported);
    SET_PTR(hcclGetAicpuOpStreamNotifyPtr, "HcclGetAicpuOpStreamNotify", StubHcclGetAicpuOpStreamNotify, g_hcclGetAicpuOpStreamNotifySupported);
    SET_PTR(hcclAllocComResourcePtr, "HcclAllocComResource", StubHcclAllocComResource, g_hcclAllocComResourceSupported);
    SET_PTR(hcclAllocComResourceByTilingPtr, "HcclAllocComResourceByTiling", StubHcclAllocComResourceByTiling, g_hcclAllocComResourceByTilingSupported);
    SET_PTR(hcclGetAicpuOpStreamAndNotifyPtr, "HcclGetAicpuOpStreamAndNotify", StubHcclGetAicpuOpStreamAndNotify, g_hcclGetAicpuOpStreamAndNotifySupported);
    SET_PTR(hcclGetTopoDescPtr, "HcclGetTopoDesc", StubHcclGetTopoDesc, g_hcclGetTopoDescSupported);
    SET_PTR(hcclCommRegisterPtr, "HcclCommRegister", StubHcclCommRegister, g_hcclCommRegisterSupported);
    SET_PTR(hcclCommDeregisterPtr, "HcclCommDeregister", StubHcclCommDeregister, g_hcclCommDeregisterSupported);
    SET_PTR(hcclCommExchangeMemPtr, "HcclCommExchangeMem", StubHcclCommExchangeMem, g_hcclCommExchangeMemSupported);

    #undef SET_PTR
}

void HcclExDlFini(void) {
    hcclCreateComResourcePtr = StubHcclCreateComResource;
    g_hcclCreateComResourceSupported = false;
    hcclGetAicpuOpStreamNotifyPtr = StubHcclGetAicpuOpStreamNotify;
    g_hcclGetAicpuOpStreamNotifySupported = false;
    hcclAllocComResourcePtr = StubHcclAllocComResource;
    g_hcclAllocComResourceSupported = false;
    hcclAllocComResourceByTilingPtr = StubHcclAllocComResourceByTiling;
    g_hcclAllocComResourceByTilingSupported = false;
    hcclGetAicpuOpStreamAndNotifyPtr = StubHcclGetAicpuOpStreamAndNotify;
    g_hcclGetAicpuOpStreamAndNotifySupported = false;
    hcclGetTopoDescPtr = StubHcclGetTopoDesc;
    g_hcclGetTopoDescSupported = false;
    hcclCommRegisterPtr = StubHcclCommRegister;
    g_hcclCommRegisterSupported = false;
    hcclCommDeregisterPtr = StubHcclCommDeregister;
    g_hcclCommDeregisterSupported = false;
    hcclCommExchangeMemPtr = StubHcclCommExchangeMem;
    g_hcclCommExchangeMemSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHcclCreateComResource(void) { return g_hcclCreateComResourceSupported; }
extern "C" bool HcommIsSupportHcclGetAicpuOpStreamNotify(void) { return g_hcclGetAicpuOpStreamNotifySupported; }
extern "C" bool HcommIsSupportHcclAllocComResource(void) { return g_hcclAllocComResourceSupported; }
extern "C" bool HcommIsSupportHcclAllocComResourceByTiling(void) { return g_hcclAllocComResourceByTilingSupported; }
extern "C" bool HcommIsSupportHcclGetAicpuOpStreamAndNotify(void) { return g_hcclGetAicpuOpStreamAndNotifySupported; }
extern "C" bool HcommIsSupportHcclGetTopoDesc(void) { return g_hcclGetTopoDescSupported; }
extern "C" bool HcommIsSupportHcclCommRegister(void) { return g_hcclCommRegisterSupported; }
extern "C" bool HcommIsSupportHcclCommDeregister(void) { return g_hcclCommDeregisterSupported; }
extern "C" bool HcommIsSupportHcclCommExchangeMem(void) { return g_hcclCommExchangeMemSupported; }