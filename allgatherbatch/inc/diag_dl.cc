#include "diag_dl.h"

#include <cstdio>
#include <dlfcn.h>
#include <pthread.h>

#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

constexpr HcclResult kHcclNotSupported = static_cast<HcclResult>(-2);

void *g_diagLibHandle = nullptr;
bool g_hcommRegOpInfoSupported = false;
bool g_hcommRegOpTaskExceptionSupported = false;
pthread_once_t g_diagDlOnce = PTHREAD_ONCE_INIT;

HcclResult StubHcommRegOpInfo(const char *commId, void *opInfo, size_t size)
{
    (void)commId;
    (void)opInfo;
    (void)size;
    HCCL_ERROR("[allgatherbatch][diag_dl] HcommRegOpInfo not supported");
    return kHcclNotSupported;
}

HcclResult StubHcommRegOpTaskException(const char *commId, HcommGetOpInfoCallback callback)
{
    (void)commId;
    (void)callback;
    HCCL_ERROR("[allgatherbatch][diag_dl] HcommRegOpTaskException not supported");
    return kHcclNotSupported;
}

template <typename T>
void SetDiagPtr(T &ptr, const char *name, T stub, bool &supported)
{
    ptr = reinterpret_cast<T>(dlsym(g_diagLibHandle, name));
    if (ptr == nullptr) {
        ptr = stub;
        supported = false;
        HCCL_DEBUG("[allgatherbatch][diag_dl] %s not supported", name);
    } else {
        supported = true;
    }
}

void ResetDiagPtrToStub()
{
    hcommRegOpInfoPtr = &StubHcommRegOpInfo;
    hcommRegOpTaskExceptionPtr = &StubHcommRegOpTaskException;
    g_hcommRegOpInfoSupported = false;
    g_hcommRegOpTaskExceptionSupported = false;
}

void InitDiagDlOnce()
{
    ResetDiagPtrToStub();

#if defined(AICPU_COMPILE)
    const char *libName = "libccl_kernel.so";
#else
    const char *libName = "libhcomm.so";
#endif

    g_diagLibHandle = dlopen(libName, RTLD_NOW);
    if (g_diagLibHandle == nullptr) {
        std::fprintf(stderr, "[allgatherbatch][diag_dl] Failed to open %s: %s\n", libName, dlerror());
        return;
    }

    AllGatherBatchDiagDlInit(g_diagLibHandle);
}

void EnsureDiagDlInit()
{
    pthread_once(&g_diagDlOnce, InitDiagDlOnce);
}

}  // namespace

HcclResult (*hcommRegOpInfoPtr)(const char *, void *, size_t) = &StubHcommRegOpInfo;
HcclResult (*hcommRegOpTaskExceptionPtr)(const char *, HcommGetOpInfoCallback) = &StubHcommRegOpTaskException;

void AllGatherBatchDiagDlInit(void *libHandle)
{
    g_diagLibHandle = libHandle;
    if (g_diagLibHandle == nullptr) {
        ResetDiagPtrToStub();
        return;
    }

    SetDiagPtr(hcommRegOpInfoPtr, "HcommRegOpInfo", &StubHcommRegOpInfo, g_hcommRegOpInfoSupported);
    SetDiagPtr(hcommRegOpTaskExceptionPtr,
        "HcommRegOpTaskException",
        &StubHcommRegOpTaskException,
        g_hcommRegOpTaskExceptionSupported);
}

void AllGatherBatchDiagDlFini(void)
{
    ResetDiagPtrToStub();
    if (g_diagLibHandle != nullptr) {
        dlclose(g_diagLibHandle);
        g_diagLibHandle = nullptr;
    }
}

bool HcommIsSupportHcommRegOpInfo(void)
{
    EnsureDiagDlInit();
    return g_hcommRegOpInfoSupported;
}

bool HcommIsSupportHcommRegOpTaskException(void)
{
    EnsureDiagDlInit();
    return g_hcommRegOpTaskExceptionSupported;
}

}  // namespace ops_hccl_allgatherbatch
