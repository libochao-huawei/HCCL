#include "log.h"
#include "hcom_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
static HcclResult (*hcomGetCommCCLBufferSizePtr)(const char*, uint64_t&) = NULL;
static HcclResult (*hcomGetL0TopoTypeExPtr)(const char*, CommTopo*, uint32_t) = NULL;
static HcclResult (*hcomGetRankSizeExPtr)(const char*, uint32_t*, uint32_t) = NULL;
static HcclResult (*hcomGetCommHandleByGroupPtr)(const char*, void**) = NULL;
static HcclResult (*hcomGetRankSizePtr)(const char*, uint32_t*) = NULL;

// 添加支持标志（静态，默认 false）
static bool g_hcomGetCommCCLBufferSizeSupported = false;
static bool g_hcomGetL0TopoTypeExSupported = false;
static bool g_hcomGetRankSizeExSupported = false;
static bool g_hcomGetCommHandleByGroupSupported = false;
static bool g_hcomGetRankSizeSupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcomGetCommCCLBufferSize(const char* group, uint64_t& size) {
    (void)group; (void)size;
    HCCL_ERROR("[HcclWrapper] HcomGetCommCCLBufferSize not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t flag) {
    (void)group; (void)topoType; (void)flag;
    HCCL_ERROR("[HcclWrapper] HcomGetL0TopoTypeEx not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetRankSizeEx(const char* group, uint32_t* rankSize, uint32_t flag) {
    (void)group; (void)rankSize; (void)flag;
    HCCL_ERROR("[HcclWrapper] HcomGetRankSizeEx not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetCommHandleByGroup(const char* group, void** commHandle) {
    (void)group; (void)commHandle;
    HCCL_ERROR("[HcclWrapper] HcomGetCommHandleByGroup not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetRankSize(const char* group, uint32_t* rankSize) {
    (void)group; (void)rankSize;
    HCCL_ERROR("[HcclWrapper] HcomGetRankSize not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// ---------- 初始化函数 ----------
void HcomDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) { \
                ptr = stub; \
                support_flag = false; \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcomGetCommCCLBufferSizePtr, "HcomGetCommCCLBufferSize", StubHcomGetCommCCLBufferSize, g_hcomGetCommCCLBufferSizeSupported);
    SET_PTR(hcomGetL0TopoTypeExPtr, "HcomGetL0TopoTypeEx", StubHcomGetL0TopoTypeEx, g_hcomGetL0TopoTypeExSupported);
    SET_PTR(hcomGetRankSizeExPtr, "HcomGetRankSizeEx", StubHcomGetRankSizeEx, g_hcomGetRankSizeExSupported);
    SET_PTR(hcomGetCommHandleByGroupPtr, "HcomGetCommHandleByGroup", StubHcomGetCommHandleByGroup, g_hcomGetCommHandleByGroupSupported);
    SET_PTR(hcomGetRankSizePtr, "HcomGetRankSize", StubHcomGetRankSize, g_hcomGetRankSizeSupported);

    #undef SET_PTR
}

void HcomDlFini(void) {
    hcomGetCommCCLBufferSizePtr = StubHcomGetCommCCLBufferSize;
    g_hcomGetCommCCLBufferSizeSupported = false;
    hcomGetL0TopoTypeExPtr = StubHcomGetL0TopoTypeEx;
    g_hcomGetL0TopoTypeExSupported = false;
    hcomGetRankSizeExPtr = StubHcomGetRankSizeEx;
    g_hcomGetRankSizeExSupported = false;
    hcomGetCommHandleByGroupPtr = StubHcomGetCommHandleByGroup;
    g_hcomGetCommHandleByGroupSupported = false;
    hcomGetRankSizePtr = StubHcomGetRankSize;
    g_hcomGetRankSizeSupported = false;
}

// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcomGetCommCCLBufferSize(const char* group, uint64_t& size) {
    return hcomGetCommCCLBufferSizePtr(group, size);
}
HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t flag) {
    return hcomGetL0TopoTypeExPtr(group, topoType, flag);
}
HcclResult HcomGetRankSizeEx(const char* group, uint32_t* rankSize, uint32_t flag) {
    return hcomGetRankSizeExPtr(group, rankSize, flag);
}
HcclResult HcomGetCommHandleByGroup(const char* group, void** commHandle) {
    return hcomGetCommHandleByGroupPtr(group, commHandle);
}
HcclResult HcomGetRankSize(const char* group, uint32_t* rankSize) {
    return hcomGetRankSizePtr(group, rankSize);
}

// ---------- 查询函数实现 ----------
extern "C" bool HcommIsSupportHcomGetCommCCLBufferSize(void) {
    return g_hcomGetCommCCLBufferSizeSupported;
}
extern "C" bool HcommIsSupportHcomGetL0TopoTypeEx(void) {
    return g_hcomGetL0TopoTypeExSupported;
}
extern "C" bool HcommIsSupportHcomGetRankSizeEx(void) {
    return g_hcomGetRankSizeExSupported;
}
extern "C" bool HcommIsSupportHcomGetCommHandleByGroup(void) {
    return g_hcomGetCommHandleByGroupSupported;
}
extern "C" bool HcommIsSupportHcomGetRankSize(void) {
    return g_hcomGetRankSizeSupported;
}