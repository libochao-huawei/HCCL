#include <unistd.h>
#include <vector>
#include <atomic>
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "alg_param.h"
#include <dlfcn.h>

using namespace ops_hccl;

extern "C" unsigned int HcclLaunchAicpuKernel(OpParam *param);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName,
    aclrtFuncHandle *funcHandle)
{
    // AICPU模式直掉kernel函数, 不使用funcHandle, 桩函数直接返回成功
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
    aclrtParamHandle *paramHandle)
{
    if (argsHandle == nullptr || param == nullptr || paramSize == 0 ||paramSize > sizeof(OpParam)) {
        printf("[ERROR] [aclrtKernelArgsAppend] invalid input param\n");
        return ACL_ERROR_INVALID_PARAM;
    }

    memcpy(argsHandle, param, paramSize);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    // 目前无需处理, 桩函数直接返回成功
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    if (argsHandle == nullptr) {
        printf("[ERROR] [aclrtKernelArgsInit] invalid input argsHandle\n");
        return ACL_ERROR_INVALID_PARAM;
    }

    // AICPU模式下，argsHandle大小等于OpParam结构体大小
    *argsHandle = malloc(sizeof(OpParam));
    if (*argsHandle == nullptr) {
        printf("[ERROR] [aclrtKernelArgsInit] malloc argsHandle failed\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim,
    aclrtStream stream, aclrtLaunchKernelCfg *cfg,
    aclrtArgsHandle argsHandle, void *reserve)
{
    printf("[DEBUG] [aclrtLaunchKernelWithConfig] HcclLaunchAicpuKernel start\n");
    if (argsHandle == nullptr || stream == nullptr) {
        printf("[ERROR] [aclrtLaunchKernelWithConfig] invalid input argsHandle or stream\n");
        return ACL_ERROR_INVALID_PARAM;
    }

    aclError retCode = ACL_SUCCESS;
    OpParam* param = reinterpret_cast<OpParam*>(argsHandle);
    auto ret = HcclLaunchAicpuKernel(param);
    if (ret != ACL_SUCCESS) {
        printf("[ERROR] [aclrtLaunchKernelWithConfig] HcclLaunchAicpuKernel failed\n");
        retCode = ACL_ERROR_INTERNAL_ERROR;
    }

    if (argsHandle != nullptr) {
        // argsHandle内存由aclrtKernelArgsInit开辟此处释放
        free(argsHandle);
        argsHandle = nullptr;
    }
    return retCode;
}

#ifdef __cplusplus
}
#endif  // __cplusplus

namespace ops_hccl{
    HcclResult LoadAICPUKernel(void)
    {
        printf("LoadAICPUKernel\n");
        return HCCL_SUCCESS;
    }
}
