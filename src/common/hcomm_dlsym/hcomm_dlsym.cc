#include "hcomm_dlsym.h"
#include "hccl_res_dl.h"
#include "hccl_rank_graph_dl.h"
#include "hcomm_primitives_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <acl/acl.h>

static void* gLibHandle = NULL;
static int gHcommVersion = 0;

int GetHcommVersion(void) {
    if (gHcommVersion == 0) {
        char hcommPkgName[] = "hcomm";
        if (aclsysGetVersionNum(hcommPkgName, &gHcommVersion) != ACL_SUCCESS) {
            gHcommVersion = 0;
        }
    }

    return gHcommVersion;
}

// 初始化
int HcommDlInit(void) {
    if (gLibHandle != NULL) return 0;

    gLibHandle = dlopen("libhcomm.so", RTLD_NOW);
    if (!gLibHandle) {
        fprintf(stderr, "[HcclWrapper] Failed to open libhcomm: %s\n", dlerror());
        return -1;
    }

    dlerror();

    HcclResDlInit(gLibHandle);
    HcclResDlInit(gLibHandle);
    HcommPrimitivesDlInit(gLibHandle);
    return 0;
}

void HcommDlFini(void) {
    if (gLibHandle) {
        HcclResDlFini();
        HcclResDlFini();
        HcommPrimitivesDlFini();

        dlclose(gLibHandle);
        gLibHandle = NULL;
    }
}

__attribute__((constructor)) void InitHcommDlsym()
{
    (void)HcommDlInit();
}