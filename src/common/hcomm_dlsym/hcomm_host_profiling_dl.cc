#include "log.h"
#include "hcomm_host_profiling_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcommProfilingRegThreadPtr)(HcomProInfo, ThreadHandle*) = NULL;
HcclResult (*hcommProfilingUnRegThreadPtr)(HcomProInfo, ThreadHandle*) = NULL;
HcclResult (*hcommProfilingReportKernelPtr)(uint64_t, const char*) = NULL;
HcclResult (*hcommProfilingReportOpPtr)(HcomProInfo) = NULL;
uint64_t (*hcommGetProfilingSysCycleTimePtr)() = NULL;

// 添加支持标志（静态，默认 false）
static bool g_hcommProfilingRegThreadSupported = false;
static bool g_hcommProfilingUnRegThreadSupported = false;
static bool g_hcommProfilingReportKernelSupported = false;
static bool g_hcommProfilingReportOpSupported = false;
static bool g_hcommGetProfilingSysCycleTimeSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcommProfilingRegThread(HcomProInfo profInfo, ThreadHandle* threads) {
    (void)profInfo; (void)threads;
    HCCL_ERROR("[HcclWrapper] HcommProfilingRegThread not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingUnRegThread(HcomProInfo profInfo, ThreadHandle* threads) {
    (void)profInfo; (void)threads;
    HCCL_ERROR("[HcclWrapper] HcommProfilingUnRegThread not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportKernel(uint64_t beginTime, const char* profName) {
    (void)beginTime; (void)profName;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportKernel not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportOp(HcomProInfo profInfo) {
    (void)profInfo;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportOp not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static uint64_t StubHcommGetProfilingSysCycleTime() {
    HCCL_ERROR("[HcclWrapper] HcommGetProfilingSysCycleTime not supported");
    return 0;
}

// 初始化
void HcommProfilingDlInit(void* libHcommHandle) {
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

    SET_PTR(hcommProfilingRegThreadPtr, "HcommProfilingRegThread", StubHcommProfilingRegThread, g_hcommProfilingRegThreadSupported);
    SET_PTR(hcommProfilingUnRegThreadPtr, "HcommProfilingUnRegThread", StubHcommProfilingUnRegThread, g_hcommProfilingUnRegThreadSupported);
    SET_PTR(hcommProfilingReportKernelPtr, "HcommProfilingReportKernel", StubHcommProfilingReportKernel, g_hcommProfilingReportKernelSupported);
    SET_PTR(hcommProfilingReportOpPtr, "HcommProfilingReportOp", StubHcommProfilingReportOp, g_hcommProfilingReportOpSupported);
    SET_PTR(hcommGetProfilingSysCycleTimePtr, "HcommGetProfilingSysCycleTime", StubHcommGetProfilingSysCycleTime, g_hcommGetProfilingSysCycleTimeSupported);

    #undef SET_PTR
}

void HcommProfilingDlFini(void) {
    hcommProfilingRegThreadPtr = StubHcommProfilingRegThread;
    g_hcommProfilingRegThreadSupported = false;
    hcommProfilingUnRegThreadPtr = StubHcommProfilingUnRegThread;
    g_hcommProfilingUnRegThreadSupported = false;
    hcommProfilingReportKernelPtr = StubHcommProfilingReportKernel;
    g_hcommProfilingReportKernelSupported = false;
    hcommProfilingReportOpPtr = StubHcommProfilingReportOp;
    g_hcommProfilingReportOpSupported = false;
    hcommGetProfilingSysCycleTimePtr = StubHcommGetProfilingSysCycleTime;
    g_hcommGetProfilingSysCycleTimeSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHcommProfilingRegThread(void) {
    return g_hcommProfilingRegThreadSupported;
}
extern "C" bool HcommIsSupportHcommProfilingUnRegThread(void) {
    return g_hcommProfilingUnRegThreadSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportKernel(void) {
    return g_hcommProfilingReportKernelSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportOp(void) {
    return g_hcommProfilingReportOpSupported;
}
extern "C" bool HcommIsSupportHcommGetProfilingSysCycleTime(void) {
    return g_hcommGetProfilingSysCycleTimeSupported;
}