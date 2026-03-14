#include "log.h"
#include "hcomm_diag_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcommRegOpInfoPtr)(const char*, void*, size_t) = NULL;
HcclResult (*hcommRegOpTaskExceptionPtr)(const char*, HcommGetOpInfoCallback) = NULL;

// 添加支持标志（静态，默认 false）
static bool g_hcommRegOpInfoSupported = false;
static bool g_hcommRegOpTaskExceptionSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcommRegOpInfo(const char* commId, void* opInfo, size_t size) {
    (void)commId; (void)opInfo; (void)size;
    HCCL_ERROR("[HcclWrapper] HcommRegOpInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcommRegOpTaskException(const char* commId, HcommGetOpInfoCallback callback) {
    (void)commId; (void)callback;
    HCCL_ERROR("[HcclWrapper] HcommRegOpTaskException not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcommDiagDlInit(void* libHcommHandle) {
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

    SET_PTR(hcommRegOpInfoPtr, "HcommRegOpInfo", StubHcommRegOpInfo, g_hcommRegOpInfoSupported);
    SET_PTR(hcommRegOpTaskExceptionPtr, "HcommRegOpTaskException", StubHcommRegOpTaskException, g_hcommRegOpTaskExceptionSupported);

    #undef SET_PTR
}

void HcommDiagDlFini(void) {
    hcommRegOpInfoPtr = StubHcommRegOpInfo;
    g_hcommRegOpInfoSupported = false;
    hcommRegOpTaskExceptionPtr = StubHcommRegOpTaskException;
    g_hcommRegOpTaskExceptionSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHcommRegOpInfo(void) {
    return g_hcommRegOpInfoSupported;
}
extern "C" bool HcommIsSupportHcommRegOpTaskException(void) {
    return g_hcommRegOpTaskExceptionSupported;
}