/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <vector>
#include <atomic>
#include <set>
#include <chrono>
#include <ctime>
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "hccl_shm_pub.h"
#include "hccl_vm_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct ArgsBuffer {
    void *data;
    uint64_t size;
};

namespace sim {
constexpr size_t MAX_ARGS_BUFF_SIZE = 64 * 1024U;

struct FuncArgsDetail {
    uint8_t *argsData{nullptr};
    size_t argsDataSize{0};
    bool  isHold{false};
};

struct FuncArgs
{
    uint8_t *argsBuff{nullptr};
    size_t argsBufferSize{0};
    bool isSysMem{true};
    size_t useOffset{0};
    std::vector<FuncArgsDetail> argDetail;

    FuncArgs()
    {
        argsBuff = new uint8_t[MAX_ARGS_BUFF_SIZE];
        argsBufferSize = MAX_ARGS_BUFF_SIZE;
        isSysMem = true;
    }
    ~FuncArgs()
    {
        if (isSysMem && argsBuff) {
            delete[] argsBuff;
        }
    }

    void ResetArgsBuff()
    {
        if (isSysMem && argsBuff) {
            delete[] argsBuff;
            argsBuff = nullptr;
            argsBufferSize = 0;
        }
    }
};

struct FuncHandle
{
    std::string funcName{""};
    std::string kernelName{""};
    std::vector<FuncArgs> funArgs;
};

struct DevBinary;
struct Program
{
    DevBinary* bin{nullptr};
    std::map<std::string, FuncHandle> funcs;
};

struct DevBinary
{
    std::string binPath{""};
    void* data{nullptr};
    size_t dataLen{0};
    Program prog;
};

std::set<DevBinary*> g_kernelBinary;
}

aclrtBinary aclrtCreateBinary(const void *data, size_t dataLen)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return 0;

    /////////////////////////////////////////////////////////////////////////
    sim::DevBinary* binPtr = new sim::DevBinary();
 
    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("[aclrtBinaryLoadFromFile] failed");
        return 0;
    }

    binPtr->data = (void*)new char[dataLen];
    memcpy(binPtr->data, data, dataLen);
    binPtr->dataLen = dataLen;
    return (aclrtBinary)(binPtr);
}

aclError aclrtDestroyBinary(aclrtBinary binary)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
    ///////////////////////////////////////////////////////

    sim::DevBinary* binPtr = (sim::DevBinary*)binary;
    if (auto search = sim::g_kernelBinary.find(binPtr); search == sim::g_kernelBinary.end()){
        HCCL_VM_ERROR("[aclrtDestroyBinary] can not find this binary");
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    sim::g_kernelBinary.erase(binPtr);

    if (binPtr->data != nullptr) {
        delete [] (char *)binPtr->data;
    }

    delete binPtr;

    return ACL_SUCCESS;
}

aclError aclrtBinaryLoad(const aclrtBinary binary, aclrtBinHandle *binHandle)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
    ///////////////////////////////////////////////////////

    sim::DevBinary* binPtr = (sim::DevBinary*)binary;

    // TODO 需要解析binary

    *binHandle = (aclrtBinHandle)&(binPtr->prog);
    return ACL_SUCCESS;
}

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    return ACL_SUCCESS;

    ///////////////////////////////////////////////////////////////////////////
    sim::DevBinary* binPtr = new sim::DevBinary();
    binPtr->binPath = binPath;
    binPtr->prog.bin = binPtr;

    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("[aclrtBinaryLoadFromFile] file:{} insert failed", binPath);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    *binHandle = (aclrtBinHandle)&(binPtr->prog);

    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromData(const void *data, size_t length, const aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    return ACL_SUCCESS;

}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    return ACL_SUCCESS;

    ///////////////////////////////////////////////////////////////////////////
    sim::Program* prog = (sim::Program*)(uintptr_t)binHandle;

    std::string funcName(kernelName);
    auto funcIter = prog->funcs.find(funcName);
    if (funcIter == prog->funcs.end()) {
        HCCL_VM_ERROR("[aclrtBinaryGetFunction] kernelName:{} not register", kernelName);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    *funcHandle = (aclrtFuncHandle)&*funcIter;
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle *funcHandle)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHandle, void **aicAddr, void **aivAddr)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    ///////////////////////////////////////////////////////////////////////////
    sim::FuncHandle* funcHandlePtr = (sim::FuncHandle *)(uintptr_t)funcHandle;

    memcpy(name, funcHandlePtr->funcName.data(), funcHandlePtr->funcName.length());
    return ACL_SUCCESS;
}

aclError aclrtRegisterCpuFunc(const aclrtBinHandle handle, const char *funcName, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    return ACL_SUCCESS;
    ///////////////////////////////////////////////////////////////////////////
    sim::Program* prog = (sim::Program*)(uintptr_t)handle;

    sim::FuncHandle func;
    func.funcName = funcName;
    func.kernelName = kernelName;
    auto res = prog->funcs.insert(std::pair<std::string, sim::FuncHandle>(func.funcName, func));
    if (!res.second) {
        HCCL_VM_ERROR("[aclrtRegisterCpuFunc] func:{} kernelName:{} insert failed", funcName, kernelName);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    *funcHandle = (aclrtFuncHandle)&*res.first;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    if (argsHandle == nullptr) {
        HCCL_VM_ERROR("[aclrtKernelArgsInit] invalid input argsHandle");
        return ACL_ERROR_INVALID_PARAM;
    }

    ArgsBuffer *buffer = (ArgsBuffer *)malloc(sizeof(ArgsBuffer));
    if (buffer == nullptr) {
        HCCL_VM_ERROR("[aclrtKernelArgsInit] malloc ArgsBuffer failed");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    buffer->data = nullptr;
    buffer->size = 0;
    *argsHandle = reinterpret_cast<void *>(buffer);
    return ACL_SUCCESS;
    ///////////////////////////////////////////////////////////////////////////
    sim::FuncHandle* func = (sim::FuncHandle*)(uintptr_t)funcHandle;
    sim::FuncArgs args;
    func->funArgs.push_back(args);

    auto iter = func->funArgs.rbegin();
    *argsHandle = (aclrtArgsHandle)&*iter;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInitByUserMem(aclrtFuncHandle funcHandle, aclrtArgsHandle argsHandle, void *userHostMem, size_t actualArgsSize)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    return ACL_SUCCESS;
    ///////////////////////////////////////////////////////////////////////////
    sim::FuncArgs* args = (sim::FuncArgs*)argsHandle;
    args->ResetArgsBuff();
    args->argsBuff = (uint8_t*)userHostMem;
    args->argsBufferSize = actualArgsSize;
    args->isSysMem = false;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHandle, size_t userArgsSize, size_t *actualArgsSize)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    // 
    *actualArgsSize = userArgsSize;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHandle, size_t *memSize)
{
    HCCL_VM_DEBUG("[{}] not support.", __func__);
    // 句柄 + 参数的内存大小
    *memSize = sim::MAX_ARGS_BUFF_SIZE;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
    aclrtParamHandle *paramHandle)
{
    if (argsHandle == nullptr || param == nullptr || paramSize == 0) {
        HCCL_VM_ERROR("[aclrtKernelArgsAppend] invalid input param");
        return ACL_ERROR_INVALID_PARAM;
    }

    // 按照OpParam的真实大小paramSize分配内存
    ArgsBuffer *buffer = reinterpret_cast<ArgsBuffer *>(argsHandle);
    buffer->data = malloc(paramSize);
    buffer->size = paramSize;
    if (buffer->data == nullptr) {
        HCCL_VM_ERROR("[aclrtKernelArgsAppend] malloc buffer->data failed");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    memcpy(buffer->data, param, paramSize);
    return ACL_SUCCESS;
    ///////////////////////////////////////////////////////////
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail argsDetail{};
    argsDetail.argsData = args->argsBuff + args->useOffset;
    argsDetail.argsDataSize = paramSize;

    memcpy(argsDetail.argsData, param, paramSize);
    args->useOffset += paramSize;
    args->argDetail.push_back(argsDetail);
    auto iter = args->argDetail.rbegin();
    *paramHandle = (aclrtParamHandle)&*iter;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle, aclrtParamHandle *paramHandle)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
    ///////////////////////////////////////////////////////////
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail argsDetail{};
    argsDetail.isHold = true;

    args->argDetail.push_back(argsDetail);
    auto iter = args->argDetail.rbegin();
    *paramHandle = (aclrtParamHandle)&*iter;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, size_t dataSize, void **bufferAddr)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
    ////////////////////////////////////////////////////
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    detail->argsData = args->argsBuff + args->useOffset;
    detail->argsDataSize = dataSize;
    *bufferAddr = (void*)detail->argsData;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsParaUpdate(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, void *param, size_t paramSize)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
////////////////////////////////////////////////////
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    if (detail->isHold || detail->argsDataSize != paramSize) {
        HCCL_VM_ERROR("[aclrtKernelArgsParaUpdate] invalid param handle type:{}", detail->isHold);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    memcpy(detail->argsData, param, paramSize);

    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernel(aclrtFuncHandle funcHandle, uint32_t blockDim, const void *argsData, size_t argsSize, aclrtStream stream)
{
    HCCL_VM_DEBUG("[{}] stub not support.", __func__);
    
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, aclrtLaunchKernelCfg *cfg, aclrtArgsHandle argsHandle, void *reserve)
{
    HCCL_VM_DEBUG("[aclrtLaunchKernelWithConfig] HcclLaunchAicpuKernel start");
    if (argsHandle == nullptr || stream == nullptr) {
        HCCL_VM_ERROR("[ERROR] [aclrtLaunchKernelWithConfig] invalid input argsHandle or stream");
        return ACL_ERROR_INVALID_PARAM;
    }

#if 0
    ArgsBuffer *buffer = reinterpret_cast<ArgsBuffer *>(argsHandle);
    OpParam *param = reinterpret_cast<OpParam *>(buffer->data);
    
    void* realPtr = nullptr;
    uint64_t offsetPtr = 0;
    sim::shm::AllocatePhy(&realPtr, &offsetPtr, buffer->size);
    memcpy(realPtr, buffer->data,  buffer->size);
    if (argsHandle != nullptr) {
        free(buffer->data);
        buffer->data = nullptr;
        free(argsHandle);
        argsHandle = nullptr;
    }

    uint64_t starPtr = (uint64_t)(uintptr_t)opParam->resCtx;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [starPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr == starPtr;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclstub][aclrtReleaseMemAddress] can not find the vir ptr");
        return -1;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[ERROR][aclrtMapMem] cannot find phy Mem offset:{}", phyMemId);
        return -1;
    }

    auto resOffsetPtr = phyMemRes->addr;
    opParam->resCtx = (void*)resOffsetPtr;

    // 启动子进程
    pid_t pid = fork();
    if (pid == -1) {
        HCCL_VM_ERROR("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // 子进程
        if (setenv("LD_PRELOAD", "../lib/libhccl_proxy.so", 1) == -1) {
            HCCL_VM_ERROR("set LD_PRELOAD failed");
            exit(EXIT_FAILURE);
        }
        execlp("./device_vir",
            "device_vir",
            "libhccl_aicpu_kernel.so",
            "HcclLaunchAicpuKernel",
            std::to_string(offsetPtr).c_str());
        HCCL_VM_ERROR("execlp finished");
        exit(EXIT_FAILURE);
    } else {
        int status;
        waitpid(pid, &status, 0);
        int exitStatus = WEXITSTATUS(status);
        HCCL_VM_DEBUG("finished with status {}", exitStatus);
    }

    sim::shm::DeallocatePhy(realPtr, buffer->size);
#endif
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus