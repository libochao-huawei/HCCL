#include "profiling.h"

#include <pthread.h>
#include <dlfcn.h>
#include <cstdio>

#include "log.h"

namespace ops_hccl_allgatherbatch {

uint64_t (*hcommGetProfilingSysCycleTimePtr)() = nullptr;
HcclResult (*hcommProfilingReportOpPtr)(HcomProInfoTmp) = nullptr;
HcclResult (*hcommProfilingReportKernelPtr)(uint64_t, const char *) = nullptr;
HcclResult (*hcommProfilingRegThreadPtr)(HcomProInfoTmp, ThreadHandle *) = nullptr;
HcclResult (*hcommProfilingInitPtr)(ThreadHandle *, uint32_t) = nullptr;
HcclResult (*hcommProfilingReportMainStreamAndFirstTaskPtr)(ThreadHandle) = nullptr;
HcclResult (*hcommProfilingReportMainStreamAndLastTaskPtr)(ThreadHandle) = nullptr;
HcclResult (*hcommProfilingReportDeviceHcclOpInfoPtr)(HcomProInfoTmp) = nullptr;
HcclResult (*hcommProfilingEndPtr)(ThreadHandle *, uint32_t) = nullptr;

namespace {

constexpr HcclResult kHcclNotSupported = static_cast<HcclResult>(-2);

void *g_libHandle = nullptr;
bool g_hcommGetProfilingSysCycleTimeSupported = false;
bool g_hcommProfilingReportOpSupported = false;
bool g_hcommProfilingReportKernelSupported = false;
bool g_hcommProfilingRegThreadSupported = false;
bool g_hcommProfilingInitSupported = false;
bool g_hcommProfilingReportMainStreamAndFirstTaskSupported = false;
bool g_hcommProfilingReportMainStreamAndLastTaskSupported = false;
bool g_hcommProfilingReportDeviceHcclOpInfoSupported = false;
bool g_hcommProfilingEndSupported = false;

uint64_t StubHcommGetProfilingSysCycleTime()
{
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommGetProfilingSysCycleTime not supported");
    return 0;
}

HcclResult StubHcommProfilingReportOp(HcomProInfoTmp profInfo)
{
    (void)profInfo;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingReportOp not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommProfilingReportKernel(uint64_t beginTime, const char *profName)
{
    (void)beginTime;
    (void)profName;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingReportKernel not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommProfilingRegThread(HcomProInfoTmp profInfo, ThreadHandle *threads)
{
    (void)profInfo;
    (void)threads;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingRegThread not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommProfilingInit(ThreadHandle *threads, uint32_t threadNum)
{
    (void)threads;
    (void)threadNum;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingInit not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread)
{
    (void)thread;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingReportMainStreamAndFirstTask not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommProfilingReportMainStreamAndLastTask(ThreadHandle thread)
{
    (void)thread;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingReportMainStreamAndLastTask not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommProfilingReportDeviceHcclOpInfo(HcomProInfoTmp profInfo)
{
    (void)profInfo;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingReportDeviceHcclOpInfo not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommProfilingEnd(ThreadHandle *threads, uint32_t threadNum)
{
    (void)threads;
    (void)threadNum;
    HCCL_WARNING("[allgatherbatch][profiling_dl] HcommProfilingEnd not supported");
    return kHcclNotSupported;
}

template <typename T>
void SetProfilingPtr(
    T &ptr,
    const char *name,
    T stub,
    bool &supported)
{
    ptr = reinterpret_cast<T>(dlsym(g_libHandle, name));
    if (ptr == nullptr) {
        ptr = stub;
        supported = false;
        HCCL_DEBUG("[allgatherbatch][profiling_dl] %s not supported", name);
    } else {
        supported = true;
    }
}

void ResetToStub()
{
    hcommGetProfilingSysCycleTimePtr = &StubHcommGetProfilingSysCycleTime;
    hcommProfilingReportOpPtr = &StubHcommProfilingReportOp;
    hcommProfilingReportKernelPtr = &StubHcommProfilingReportKernel;
    hcommProfilingRegThreadPtr = &StubHcommProfilingRegThread;
    hcommProfilingInitPtr = &StubHcommProfilingInit;
    hcommProfilingReportMainStreamAndFirstTaskPtr = &StubHcommProfilingReportMainStreamAndFirstTask;
    hcommProfilingReportMainStreamAndLastTaskPtr = &StubHcommProfilingReportMainStreamAndLastTask;
    hcommProfilingReportDeviceHcclOpInfoPtr = &StubHcommProfilingReportDeviceHcclOpInfo;
    hcommProfilingEndPtr = &StubHcommProfilingEnd;
}

void InitProfilingDl()
{
    if (g_libHandle != nullptr) {
        return;
    }

    ResetToStub();

#if defined(HOST_COMPILE)
    const char *libName = "libhcomm.so";
#elif defined(AICPU_COMPILE)
    const char *libName = "libccl_kernel.so";
#else
    const char *libName = nullptr;
#endif

    if (libName == nullptr) {
        return;
    }

    g_libHandle = dlopen(libName, RTLD_NOW);
    if (g_libHandle == nullptr) {
        std::fprintf(stderr, "[allgatherbatch][profiling_dl] Failed to open %s: %s\n", libName, dlerror());
        return;
    }

    dlerror();

    SetProfilingPtr(
        hcommGetProfilingSysCycleTimePtr,
        "HcommGetProfilingSysCycleTime",
        &StubHcommGetProfilingSysCycleTime,
        g_hcommGetProfilingSysCycleTimeSupported);
    SetProfilingPtr(
        hcommProfilingReportOpPtr,
        "HcommProfilingReportOp",
        &StubHcommProfilingReportOp,
        g_hcommProfilingReportOpSupported);
    SetProfilingPtr(
        hcommProfilingReportKernelPtr,
        "HcommProfilingReportKernel",
        &StubHcommProfilingReportKernel,
        g_hcommProfilingReportKernelSupported);
    SetProfilingPtr(
        hcommProfilingRegThreadPtr,
        "HcommProfilingRegThread",
        &StubHcommProfilingRegThread,
        g_hcommProfilingRegThreadSupported);
    SetProfilingPtr(
        hcommProfilingInitPtr,
        "HcommProfilingInit",
        &StubHcommProfilingInit,
        g_hcommProfilingInitSupported);
    SetProfilingPtr(
        hcommProfilingReportMainStreamAndFirstTaskPtr,
        "HcommProfilingReportMainStreamAndFirstTask",
        &StubHcommProfilingReportMainStreamAndFirstTask,
        g_hcommProfilingReportMainStreamAndFirstTaskSupported);
    SetProfilingPtr(
        hcommProfilingReportMainStreamAndLastTaskPtr,
        "HcommProfilingReportMainStreamAndLastTask",
        &StubHcommProfilingReportMainStreamAndLastTask,
        g_hcommProfilingReportMainStreamAndLastTaskSupported);
    SetProfilingPtr(
        hcommProfilingReportDeviceHcclOpInfoPtr,
        "HcommProfilingReportDeviceHcclOpInfo",
        &StubHcommProfilingReportDeviceHcclOpInfo,
        g_hcommProfilingReportDeviceHcclOpInfoSupported);
    SetProfilingPtr(
        hcommProfilingEndPtr,
        "HcommProfilingEnd",
        &StubHcommProfilingEnd,
        g_hcommProfilingEndSupported);
}

}  // namespace

extern "C" bool HcommIsSupportHcommGetProfilingSysCycleTime(void)
{
    return g_hcommGetProfilingSysCycleTimeSupported;
}

extern "C" bool HcommIsSupportHcommProfilingReportOp(void)
{
    return g_hcommProfilingReportOpSupported;
}

extern "C" bool HcommIsSupportHcommProfilingReportKernel(void)
{
    return g_hcommProfilingReportKernelSupported;
}

extern "C" bool HcommIsSupportHcommProfilingRegThread(void)
{
    return g_hcommProfilingRegThreadSupported;
}

extern "C" bool HcommIsSupportHcommProfilingInit(void)
{
    return g_hcommProfilingInitSupported;
}

extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void)
{
    return g_hcommProfilingReportMainStreamAndFirstTaskSupported;
}

extern "C" bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void)
{
    return g_hcommProfilingReportMainStreamAndLastTaskSupported;
}

extern "C" bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void)
{
    return g_hcommProfilingReportDeviceHcclOpInfoSupported;
}

extern "C" bool HcommIsSupportHcommProfilingEnd(void)
{
    return g_hcommProfilingEndSupported;
}

__attribute__((constructor)) void InitAllGatherBatchProfilingDl()
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, InitProfilingDl);
}

}  // namespace ops_hccl_allgatherbatch
