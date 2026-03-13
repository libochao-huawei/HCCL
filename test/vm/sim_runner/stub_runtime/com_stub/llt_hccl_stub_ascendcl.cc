#include "acl/acl_rt.h"
#include "../rts_stub/rts_stub.h"
#include "hccl_sim_aicpu_stub.h"

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char *binPath, aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    CHK_PTR_NULL(kernelName);
    CHK_PTR_NULL(funcHandle);
    *funcHandle = reinterpret_cast<void*>(const_cast<char *>(kernelName));
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize, aclrtParamHandle *paramHandle)
{
    CHK_PTR_NULL(argsHandle);
    KFCTaskCommStub *task = reinterpret_cast<KFCTaskCommStub*>(argsHandle);
    KFCTaskCommStub *paramPtr = reinterpret_cast<KFCTaskCommStub*>(param);
    HCCL_INFO("[aclrtKernelArgsAppend]context[%llu]", paramPtr->context);
    task->context = paramPtr->context;
    HCCL_INFO("[aclrtKernelArgsAppend]task context[%llu]", task->context);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    CHK_PTR_NULL(argsHandle);
    char *task = new char[sizeof(KFCTaskCommStub)];
    CHK_PTR_NULL(task);
    CHK_SAFETY_FUNC_RET(memset_s(task, sizeof(KFCTaskCommStub), 0, sizeof(KFCTaskCommStub)));
    *argsHandle = reinterpret_cast<aclrtParamHandle>(task);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle, aclrtParamHandle *paramHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetPlaceHolderBuffer(
    aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, size_t dataSize, void **bufferAddr)
{
    CHK_PTR_NULL(argsHandle);
    KFCTaskCommStub *task = reinterpret_cast<KFCTaskCommStub*>(argsHandle);
    if (dataSize > 0) {
        *bufferAddr = new char[dataSize];
        if (*bufferAddr == nullptr) {
            HCCL_ERROR("malloc mem failed, dataSize[%llu]", dataSize);
            return ACL_ERROR_BAD_ALLOC;
        }
    }
    task->tilingData = reinterpret_cast<u64>(*bufferAddr);

    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
    aclrtLaunchKernelCfg *cfg, aclrtArgsHandle argsHandle, void *reserve)
{
    CHK_PTR_NULL(argsHandle);
    rtAicpuArgsEx_t argsInfo;
    static hccl::DeviceMem args = hccl::DeviceMem::alloc((sizeof(KFCTaskCommStub) + 64)); // 64:kernel name
    argsInfo.args = args.ptr();
    (void)memcpy(args.ptr(), argsHandle, sizeof(KFCTaskCommStub));
    std::string kernelName(reinterpret_cast<char*>(funcHandle));
    char *kernelNamePtr = reinterpret_cast<char*>(args.ptr()) + sizeof(KFCTaskCommStub);
    (void)strcpy(kernelNamePtr, kernelName.c_str());
    argsInfo.kernelNameAddrOffset = sizeof(KFCTaskCommStub);
    KFCTaskCommStub *task = reinterpret_cast<KFCTaskCommStub*>(argsHandle);
    HCCL_INFO("[aclrtLaunchKernelWithConfig]context[%llu]", task->context);
    rtError_t ret = rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, reinterpret_cast<char*>(funcHandle), blockDim,
        &argsInfo, nullptr, stream, RT_KERNEL_USE_SPECIAL_TIMEOUT);
    if (task->tilingData != 0) {
        delete[] reinterpret_cast<char*>(task->tilingData);
    }
    delete[] reinterpret_cast<char*>(argsHandle);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[aclrtLaunchKernelWithConfig]rtAicpuKernelLaunchExWithArgs failed, ret[%d]", ret);
        return ACL_ERROR_RT_FAILURE;
    }

    return ACL_SUCCESS;
}

aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t *value)
{
    *value = 48;
    return ACL_SUCCESS;
}