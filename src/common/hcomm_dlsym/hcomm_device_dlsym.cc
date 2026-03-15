#include "hcomm_dlsym.h"
#include "hccl_res_dl.h"
#include "hccl_rank_graph_dl.h"
#include "hcomm_primitives_dl.h"
#include "hcomm_device_profiling_dl.h"
#include "hcomm_diag_dl.h"
#include "dtype_common_dl.h"
#include <pthread.h>
#include <pthread.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void* gLibHandle = NULL;

// 初始化
int HcommDeviceDlInit(void) {
    if (gLibHandle != NULL) return 0;

    gLibHandle = dlopen("libccl_kernel.so", RTLD_NOW);
    if (!gLibHandle) {
        fprintf(stderr, "[HcclWrapper] Failed to open libccl_kernel.so: %s\n", dlerror());
        return -1;
    }

    dlerror();

    HcommPrimitivesDlInit(gLibHandle);
    HcommDeviceProfilingDlInit(gLibHandle);
    HcommDiagDlInit(gLibHandle);
    DtypeCommonDlInit(gLibHandle);
    return 0;
}

void HcommDeviceDlFini(void) {
    if (gLibHandle) {
        HcommPrimitivesDlFini();
        HcommDeviceProfilingDlFini();
        HcommDiagDlFini();
        DtypeCommonDlFini();

        dlclose(gLibHandle);
        gLibHandle = NULL;
    }
}

__attribute__((constructor)) void InitHcommDeviceDlsym()
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, (void)HcommDeviceDlInit);
}