#include "hcomm_dlsym.h"
#include "hccl_res_dl.h"
#include "hccl_rank_graph_dl.h"
#include "hcomm_primitives_dl.h"
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
    return 0;
}

void HcommDeviceDlFini(void) {
    if (gLibHandle) {
        HcommPrimitivesDlFini(gLibHandle);

        dlclose(gLibHandle);
        gLibHandle = NULL;
    }
}

__attribute__((constructor)) void InitHcommDeviceDlsym()
{
    (void)HcommDeviceDlInit();
}