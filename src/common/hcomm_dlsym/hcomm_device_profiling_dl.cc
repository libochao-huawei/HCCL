#include "log.h"
#include "hcomm_device_profiling_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcommProfilingReportMainStreamAndFirstTaskPtr)(ThreadHandle) = NULL;
HcclResult (*hcommProfilingReportMainStreamAndLastTaskPtr)(ThreadHandle) = NULL;
HcclResult (*hcommProfilingReportDeviceHcclOpInfoPtr)(HcomProInfo) = NULL;
HcclResult (*hcommProfilingInitPtr)(ThreadHandle*, uint32_t) = NULL;
HcclResult (*hcommProfilingEndPtr)(ThreadHandle*, uint32_t) = NULL;

// 添加支持标志（静态，默认 false）
static bool g_hcommProfilingReportMainStreamAndFirstTaskSupported = false;
static bool g_hcommProfilingReportMainStreamAndLastTaskSupported = false;
static bool g_hcommProfilingReportDeviceHcclOpInfoSupported = false;
static bool g_hcommProfilingInitSupported = false;
static bool g_hcommProfilingEndSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportMainStreamAndFirstTask not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportMainStreamAndLastTask(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportMainStreamAndLastTask not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingReportDeviceHcclOpInfo(HcomProInfo profInfo) {
    (void)profInfo;
    HCCL_ERROR("[HcclWrapper] HcommProfilingReportDeviceHcclOpInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingInit(ThreadHandle* threads, uint32_t threadNum) {
    (void)threads; (void)threadNum;
    HCCL_ERROR("[HcclWrapper] HcommProfilingInit not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcommProfilingEnd(ThreadHandle* threads, uint32_t threadNum) {
    (void)threads; (void)threadNum;
    HCCL_ERROR("[HcclWrapper] HcommProfilingEnd not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcommDeviceProfilingDlInit(void* libHcommHandle) {
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

    SET_PTR(hcommProfilingReportMainStreamAndFirstTaskPtr, "HcommProfilingReportMainStreamAndFirstTask",
            StubHcommProfilingReportMainStreamAndFirstTask, g_hcommProfilingReportMainStreamAndFirstTaskSupported);
    SET_PTR(hcommProfilingReportMainStreamAndLastTaskPtr, "HcommProfilingReportMainStreamAndLastTask",
            StubHcommProfilingReportMainStreamAndLastTask, g_hcommProfilingReportMainStreamAndLastTaskSupported);
    SET_PTR(hcommProfilingReportDeviceHcclOpInfoPtr, "HcommProfilingReportDeviceHcclOpInfo",
            StubHcommProfilingReportDeviceHcclOpInfo, g_hcommProfilingReportDeviceHcclOpInfoSupported);
    SET_PTR(hcommProfilingInitPtr, "HcommProfilingInit",
            StubHcommProfilingInit, g_hcommProfilingInitSupported);
    SET_PTR(hcommProfilingEndPtr, "HcommProfilingEnd",
            StubHcommProfilingEnd, g_hcommProfilingEndSupported);

    #undef SET_PTR
}

void HcommDeviceProfilingDlFini(void) {
    hcommProfilingReportMainStreamAndFirstTaskPtr = StubHcommProfilingReportMainStreamAndFirstTask;
    g_hcommProfilingReportMainStreamAndFirstTaskSupported = false;
    hcommProfilingReportMainStreamAndLastTaskPtr = StubHcommProfilingReportMainStreamAndLastTask;
    g_hcommProfilingReportMainStreamAndLastTaskSupported = false;
    hcommProfilingReportDeviceHcclOpInfoPtr = StubHcommProfilingReportDeviceHcclOpInfo;
    g_hcommProfilingReportDeviceHcclOpInfoSupported = false;
    hcommProfilingInitPtr = StubHcommProfilingInit;
    g_hcommProfilingInitSupported = false;
    hcommProfilingEndPtr = StubHcommProfilingEnd;
    g_hcommProfilingEndSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void) {
    return g_hcommProfilingReportMainStreamAndFirstTaskSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void) {
    return g_hcommProfilingReportMainStreamAndLastTaskSupported;
}
extern "C" bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void) {
    return g_hcommProfilingReportDeviceHcclOpInfoSupported;
}
extern "C" bool HcommIsSupportHcommProfilingInit(void) {
    return g_hcommProfilingInitSupported;
}
extern "C" bool HcommIsSupportHcommProfilingEnd(void) {
    return g_hcommProfilingEndSupported;
}