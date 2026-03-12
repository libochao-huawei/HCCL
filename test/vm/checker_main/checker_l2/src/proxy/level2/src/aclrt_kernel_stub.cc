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
    std::vector<FuncArgsDetail*> argDetail;

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

        for (auto& arg : argDetail) {
            delete arg;
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
    std::vector<FuncArgs*> funArgs;
    ~FuncHandle() {
        for (auto& funcArg : funArgs) {
            delete funcArg;
        }
    }
};

struct DevBinary;
struct Program
{
    DevBinary* bin{nullptr};
    std::map<std::string, FuncHandle*> funcs;
    ~Program()
    {
        for (auto& func : funcs) {
            delete func.second;
        }
    }
};

struct DevBinary
{
    std::string binPath{""};
    void* data{nullptr};
    size_t dataLen{0};
    Program prog;

    ~DevBinary()
    {
        if (data != nullptr) {
            delete [] (char *)data;
        }
    }
};

std::set<DevBinary*> g_kernelBinary;
}

aclrtBinary aclrtCreateBinary(const void *data, size_t dataLen)
{
    sim::DevBinary* binPtr = new sim::DevBinary();
 
    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("failed");
        return 0;
    }

    binPtr->data = (void*)new char[dataLen];
    memcpy(binPtr->data, data, dataLen);
    binPtr->dataLen = dataLen;
    HCCL_VM_INFO("[KERNEL] stub dataLen{:d} binary{:p}", dataLen, (aclrtBinary)(binPtr));
    return (aclrtBinary)(binPtr);
}

aclError aclrtDestroyBinary(aclrtBinary binary)
{
    sim::DevBinary* binPtr = (sim::DevBinary*)binary;
    if (auto search = sim::g_kernelBinary.find(binPtr); search == sim::g_kernelBinary.end()){
        HCCL_VM_ERROR("[aclrtDestroyBinary] can not find this binary");
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    sim::g_kernelBinary.erase(binPtr);

    HCCL_VM_DEBUG("[KERNEL] stub  binPtr{:p}", binary);
    delete binPtr;
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoad(const aclrtBinary binary, aclrtBinHandle *binHandle)
{
    sim::DevBinary* binPtr = (sim::DevBinary*)binary;
    // TODO 需要解析binary
    *binHandle = (aclrtBinHandle)&(binPtr->prog);
    HCCL_VM_DEBUG("[KERNEL] stub  binHandle:{:p}", *binHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    HCCL_VM_DEBUG("[KERNEL] stub not support.");
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    sim::DevBinary* binPtr = new sim::DevBinary();
    binPtr->binPath = binPath;
    binPtr->prog.bin = binPtr;

    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("[aclrtBinaryLoadFromFile] file:{} insert failed", binPath);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    *binHandle = (aclrtBinHandle)&(binPtr->prog);
    HCCL_VM_DEBUG("[KERNEL] stub  binPath:{} binHandle{:p}", binPath, *binHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromData(const void *data, size_t length, const aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    HCCL_VM_DEBUG("not support.");
    return ACL_SUCCESS;

}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    sim::Program* prog = (sim::Program*)(uintptr_t)binHandle;

    std::string funcName(kernelName);
    auto funcIter = prog->funcs.find(funcName);
    if (funcIter == prog->funcs.end()) {
        HCCL_VM_WARN("kernelName:{} not register insert it", kernelName);

        sim::FuncHandle* func = new sim::FuncHandle;
        func->funcName = funcName;
        func->kernelName = kernelName;
        auto res = prog->funcs.insert(std::pair<std::string, sim::FuncHandle*>(func->funcName, func));
        if (!res.second) {
            HCCL_VM_ERROR("[aclrtRegisterCpuFunc] func:{} kernelName:{} insert failed", funcName, kernelName);
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        *funcHandle = reinterpret_cast<aclrtFuncHandle>(func);
    } else {
        *funcHandle = reinterpret_cast<aclrtFuncHandle>(funcIter->second);
    }

    HCCL_VM_DEBUG("[KERNEL] stub  funcHandle:{:p}", *funcHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle *funcHandle)
{
    HCCL_VM_DEBUG("[KERNEL] stub not support.");
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHandle, void **aicAddr, void **aivAddr)
{
    HCCL_VM_DEBUG("stub not support.");
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name)
{
    sim::FuncHandle* funcHandlePtr = (sim::FuncHandle *)(uintptr_t)funcHandle;

    memcpy(name, funcHandlePtr->funcName.data(), funcHandlePtr->funcName.length());
    HCCL_VM_DEBUG("[KERNEL] stub  funcName{}", funcHandlePtr->funcName.data());
    return ACL_SUCCESS;
}

aclError aclrtRegisterCpuFunc(const aclrtBinHandle handle, const char *funcName, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    sim::Program* prog = (sim::Program*)(uintptr_t)handle;

    sim::FuncHandle* func = new sim::FuncHandle;
    func->funcName = funcName;
    func->kernelName = kernelName;
    auto res = prog->funcs.insert(std::pair<std::string, sim::FuncHandle*>(func->funcName, func));
    if (!res.second) {
        HCCL_VM_ERROR("func:{} kernelName:{} insert failed", funcName, kernelName);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    *funcHandle = reinterpret_cast<aclrtFuncHandle>(func);
    HCCL_VM_DEBUG("[KERNEL] stub  funcHandle:{:p}", *funcHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    sim::FuncHandle* func = (sim::FuncHandle*)(uintptr_t)funcHandle;
    sim::FuncArgs* args = new sim::FuncArgs;
    func->funArgs.push_back(args);

    auto iter = func->funArgs.rbegin();
    *argsHandle = (aclrtArgsHandle)args;
    HCCL_VM_DEBUG("[KERNEL] FuncHandle:{:p} ArgsHandle:{:p} ", funcHandle, *argsHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInitByUserMem(aclrtFuncHandle funcHandle, aclrtArgsHandle argsHandle, void *userHostMem, size_t actualArgsSize)
{
    HCCL_VM_DEBUG("[KERNEL] stub  argsHandle:{:p} userHostMem:{:p},actualArgsSize:{:d}", argsHandle, userHostMem, actualArgsSize);
    sim::FuncArgs* args = (sim::FuncArgs*)argsHandle;
    args->ResetArgsBuff();
    args->argsBuff = (uint8_t*)userHostMem;
    args->argsBufferSize = actualArgsSize;
    args->isSysMem = false;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHandle, size_t userArgsSize, size_t *actualArgsSize)
{
    HCCL_VM_DEBUG("[KERNEL] userArgsSize {:d}.", userArgsSize);
    // 
    *actualArgsSize = userArgsSize;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHandle, size_t *memSize)
{
    HCCL_VM_DEBUG("[KERNEL] funcHandle:{:p} userArgsSize 64k", funcHandle);
    // 句柄 + 参数的内存大小
    *memSize = sim::MAX_ARGS_BUFF_SIZE;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
    aclrtParamHandle *paramHandle)
{
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;
    HCCL_VM_DEBUG("[KERNEL] stub  argsHandle:{:p} paramSize:{:d}", argsHandle, paramSize);
    sim::FuncArgsDetail* argsDetail = new sim::FuncArgsDetail;
    argsDetail->argsData = args->argsBuff + args->useOffset;
    argsDetail->argsDataSize = paramSize;

    memcpy(argsDetail->argsData, param, paramSize);
    args->useOffset += paramSize;
    args->argDetail.push_back(argsDetail);
    *paramHandle = (aclrtParamHandle)argsDetail;
    HCCL_VM_DEBUG("[KERNEL] argsHandle:{:p} paramHandle:{:p} ", argsHandle, *paramHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle, aclrtParamHandle *paramHandle)
{
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* argsDetail = new sim::FuncArgsDetail;
    argsDetail->isHold = true;

    args->argDetail.push_back(argsDetail);
    *paramHandle = (aclrtParamHandle)argsDetail;
    HCCL_VM_DEBUG("[KERNEL] paramHandle:{:p} ", *paramHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, size_t dataSize, void **bufferAddr)
{
    HCCL_VM_DEBUG("[KERNEL] argsHandle:{:p} ParamHandle:{:p} dataSize:{:d}", argsHandle, paramHandle, dataSize);
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    detail->argsData = args->argsBuff + args->useOffset;
    detail->argsDataSize = dataSize;
    *bufferAddr = (void*)detail->argsData;
    HCCL_VM_DEBUG("[KERNEL] paramHandle:{:p} ", *bufferAddr);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsParaUpdate(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, void *param, size_t paramSize)
{
    HCCL_VM_DEBUG("[KERNEL] argsHandle:{:p} ParamHandle:{:p} paramSize:{:d}", argsHandle, paramHandle, paramSize);
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    if (detail->isHold || detail->argsDataSize != paramSize) {
        HCCL_VM_ERROR("invalid param handle type:{:d}", detail->isHold);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    memcpy(detail->argsData, param, paramSize);
    HCCL_VM_DEBUG("[KERNEL] argsData:{:p} ", (void*)detail->argsData);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    HCCL_VM_DEBUG("[KERNEL]stub not support.");
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernel(aclrtFuncHandle funcHandle, uint32_t blockDim, const void *argsData, size_t argsSize, aclrtStream stream)
{
    HCCL_VM_DEBUG("[KERNEL] stub not support.");
    
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, aclrtLaunchKernelCfg *cfg, aclrtArgsHandle argsHandle, void *reserve)
{
    HCCL_VM_DEBUG("[KERNEL] func:{:p} stream:{:p} agrs:{:p}", (void*)funcHandle, (void*)stream, (void*)argsHandle);
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