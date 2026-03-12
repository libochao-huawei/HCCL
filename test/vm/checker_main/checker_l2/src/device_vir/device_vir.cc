#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <link.h>
#include <string>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "alg_param.h"
#include "sim_shm_memory_manager.h"
using namespace ops_hccl;
using AiCpuKernelCallback = unsigned int (*)(OpParam* param);
/*
AlgResourceCtx 内部结构
|-sizeof(AlgResourceCtx)-|-sizeof(ThreadHandle) * (resRequest.slaveThreadNum + 1)-|-sizeof(ChannelInfo)*resRequest.channels.size()-|
 */

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: xxx.so func offset_ptr \n");
        return -1;
    }

    std::string binaryName = argv[1];
    std::string funcName = argv[2];
    uint64_t offsetPtr = atol(argv[3]);

    printf("binary:%s func:%s offsetPtr:%lu\n", binaryName.data(), funcName.data(), offsetPtr);

    void *handle = dlopen(binaryName.data(), RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "[ERROR] dlopen failed: %s\n", dlerror());
        return -1;
    }

    dlerror();
    AiCpuKernelCallback symbolPtr = reinterpret_cast<AiCpuKernelCallback>(dlsym(handle, funcName.data()));
    char *err = dlerror();
    if (err) {
        printf("[ERROR] Symbol %s not found: %s\n", funcName.data(), err);
        dlclose(handle);
        return -1;
    }

    OpParam* opParam = (OpParam*)sim::shm::ShmMemoryManager::GetInstance().GetPtrFromHandle(offsetPtr);
    if (opParam == nullptr) {
        fprintf(stderr, "[ERROR]get opParam from offset ptr failed:%lu\n", offsetPtr);
        return -1;
    }

    uint64_t starPtr = (uint64_t)(uintptr_t)opParam->resCtx;
    opParam->resCtx = (AlgResourceCtx*)sim::shm::ShmMemoryManager::GetInstance().GetPtrFromHandle(starPtr);
    if (symbolPtr(opParam) != 0) {
        printf("[ERROR] func %s  execution failed\n", funcName.data());
        dlclose(handle);
        return -1;
    }

    dlclose(handle);
    return 0;
}