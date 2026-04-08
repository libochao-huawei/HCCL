/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include "mmpa_api.h"
#include "adapter_acl.h"
#include "hccl_aiv_utils.h"
#include "aiv_kernel_def.h"
#include "universal_concurrent_map.h"
#include "alg_env_config.h"

using namespace std;
using namespace ops_hccl;

namespace ops_hccl {
constexpr u32 SIG_MOVE_LEFT_BITS = 20;

constexpr u32 AIV_BUFFER_PING_PONG_FACTOR = 2;

constexpr u32 MAX_BIN_FILE_SIZE = 100 * 1024 * 1024; // 最大读取100m的bin file到string中

constexpr s32 RESET_TAIL_SYNC_TAG = 2;

static bool g_init = false;
static mutex g_mut;
static aclrtBinHandle g_binHandle;
static std::unordered_map<s8*, aclrtFuncHandle> g_aivFuncMap;
static std::unordered_map<s8*, std::string> g_aivNameMap;

thread_local std::shared_ptr<InsQueue> g_recordingQueue = nullptr;
thread_local u64 g_baseInputAddr = 0;
thread_local u64 g_baseOutputAddr = 0;
static HcclResult GetMinAndMaxNpuSchedTimeOut(u64 &minNpuSchedTimeout, u64 &maxNpuSchedTimeout)
{
    uint64_t interval = 0;
    aclError aclRet = aclrtGetOpTimeOutInterval(&interval);
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_WARNING("[GetMinAndMaxNpuSchedTimeOut] get timeout interval failed, ret[%d].",
        aclRet), HCCL_E_RUNTIME);

    constexpr u64 MAX_INTERVAL = 254;
    minNpuSchedTimeout = interval;
    maxNpuSchedTimeout = MAX_INTERVAL * interval;
    return HCCL_SUCCESS;
}

static u32 GetAivTimeout()
{
    double execTimeOut = 0;
    if (!GetExternalInputExecTimeout(execTimeOut)) {
        return CUSTOM_TIMEOUT * TIME_S_TO_US;
    }
    
    u32 timeout = CUSTOM_TIMEOUT * TIME_S_TO_US;
    double timeoutUs = execTimeOut * TIME_S_TO_US;
    if (timeoutUs > static_cast<double>(std::numeric_limits<s32>::max())) {
        HCCL_WARNING("[GetAivTimeout]Get input timeout[%.2f] is out of valid range.", timeoutUs);
        return CUSTOM_TIMEOUT * TIME_S_TO_US;
    }

    u32 timeoutUsInt = static_cast<u32>(timeoutUs);
    if (timeoutUsInt == 0) {
        timeoutUsInt = CUSTOM_TIMEOUT * TIME_S_TO_US;
    }

    u64 minNpuSchedTimeout = 0;
    u64 maxNpuSchedTimeout = 0;
    if (GetMinAndMaxNpuSchedTimeOut(minNpuSchedTimeout, maxNpuSchedTimeout) != HCCL_SUCCESS) {
        HCCL_WARNING("[GetAivTimeout] get npu sched timeout range failed, use default[%u]us.", CUSTOM_TIMEOUT * TIME_S_TO_US);
        return CUSTOM_TIMEOUT * TIME_S_TO_US;
    }

    timeout = (timeoutUsInt < minNpuSchedTimeout) ? minNpuSchedTimeout
                : (timeoutUsInt > maxNpuSchedTimeout) ? maxNpuSchedTimeout
                : timeoutUsInt;
    HCCL_INFO("[GetAivTimeout]timeout[%u]us, execTimeOut[%.2f]s, minNpuSchedTimeout[%u]us, maxNpuSchedTimeout[%u]us.",
        timeout, execTimeOut, minNpuSchedTimeout, maxNpuSchedTimeout);

    return timeout;
}

using AivExtraKernelArgs = struct AivExtraKernelArgsDef {
    const void* buffersIn; // 注册的CCLIN地址，所有卡可访问
    u64 input;
    u64 output;
    u32 rank;
    u32 rankSize;
    u64 xRankSize;
    u64 yRankSize;
    u64 zRankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    u32 tag; // 第几次调用，定时重置成1
    u64 inputSliceStride;
    u64 outputSliceStride;
    u64 repeatNum;
    u64 inputRepeatStride;
    u64 outputRepeatStride;
    bool isOpBase;
    const void* headCountMem;
    const void* tailCountMem;
    const void* addOneMem;
    u32 counterMemSize;
    bool isEnableCounter;
    ExtraArgs extraArgs;

    AivExtraKernelArgsDef(const void* buffIn, u64 input, u64 output, u32 rank,
        u32 rankSize, u64 xRankSize, u64 yRankSize, u64 zRankSize,
        u64 len, u32 dataType, u32 reduceOp, u32 root, u32 tag,
        u64 inputSliceStride, u64 outputSliceStride, u64 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride,
        bool isOpBase = true,
        const void* headCountMem = nullptr, const void* tailCountMem = nullptr, const void* addOneMem = nullptr,
        u32 counterMemSize = 0, const ExtraArgs* extraArgsPtr = nullptr)
        : buffersIn(buffIn),input(input), output(output), rank(rank), rankSize(rankSize), xRankSize(xRankSize), yRankSize(yRankSize), zRankSize(zRankSize),
        len(len) ,dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag),
        inputSliceStride(inputSliceStride), outputSliceStride(outputSliceStride), repeatNum(repeatNum), inputRepeatStride(inputRepeatStride), outputRepeatStride(outputRepeatStride),
        isOpBase(isOpBase),
        headCountMem(headCountMem), tailCountMem(tailCountMem), addOneMem(addOneMem),
        counterMemSize(counterMemSize)
    {
        if (extraArgsPtr != nullptr) {
            extraArgs = *extraArgsPtr;
        }
    }
};

HcclResult GetAivOpBinaryPath(const std::string &aivBinaryName, std::string &binaryPath)
{
    // 获取二进制文件路径
    std::string libPath;
    char *getPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, getPath);
    if (getPath != nullptr) {
        libPath = getPath;
    } else {
        libPath = "/usr/local/Ascend/cann";
        HCCL_WARNING("[GetAivOpBinaryPath]ENV:ASCEND_HOME_PATH is not set");
    }
    binaryPath = libPath + "/lib64";

    // 拼接应该加载的文件
    binaryPath += "/" + aivBinaryName;
    HCCL_INFO("[GetAivOpBinaryPath]op binary file path[%s]", binaryPath.c_str());
    return HCCL_SUCCESS;
}

s8* GetFuncKey(HcclCMDType cmdType, HcclDataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
{
    return reinterpret_cast<s8*>(
        (((static_cast<u64>(cmdType) << SIG_MOVE_LEFT_BITS) + static_cast<u64>(dataType)) << SIG_MOVE_LEFT_BITS) +
        static_cast<u64>(argsType));
}

HcclResult RegisterBinaryKernel(const char* funcName, const aclrtBinHandle binHandle, const s8* funcKey)
{
    if (funcKey == nullptr) {
        return HCCL_E_PARA;
    }

    aclrtFuncHandle funcHandle;
    aclError aclRet = aclrtBinaryGetFunction(binHandle, funcName, &funcHandle);
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[RegisterBinaryKernel]errNo[0x%016llx] get function from binary error.", aclRet),
        HCCL_E_NOT_FOUND);

    g_aivFuncMap[const_cast<s8*>(funcKey)] = funcHandle;
    g_aivNameMap[const_cast<s8*>(funcKey)] = std::string(funcName);

    return HCCL_SUCCESS;
}

HcclResult GetKernelFunc(aclrtFuncHandle& funcHandle, const s8* funcKey)
{
    if (funcKey == nullptr || g_aivFuncMap.find(const_cast<s8*>(funcKey)) == g_aivFuncMap.end()) {
        return HCCL_E_PARA;
    }
    funcHandle = g_aivFuncMap[const_cast<s8*>(funcKey)];
    return HCCL_SUCCESS;
}

// Kernel注册入口，全局只需要初始化一次
HcclResult RegisterKernel()
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        return HCCL_SUCCESS;
    }

    for (const auto& item : g_aivKernelInfoMap) {
        const HcclCMDType cmdType = item.first;
        const std::string& aivBinaryName = item.second.first;
        const std::vector<AivKernelInfo>& aivKernelInfoList = item.second.second;

        HcclResult ret;
        string binFilePath;
        ret = GetAivOpBinaryPath(aivBinaryName, binFilePath);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] get aiv op binary path failed"), HCCL_E_RUNTIME);

        ret = LoadBinaryFromFile(binFilePath.c_str(), ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, 1, g_binHandle);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] read aiv kernel bin file failed"),
            HCCL_E_RUNTIME);

        for (auto &aivKernelInfo: aivKernelInfoList) {
            ret = RegisterBinaryKernel(aivKernelInfo.kernelName, g_binHandle,
                GetFuncKey(cmdType, aivKernelInfo.dataType, aivKernelInfo.argsType));
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] register binary kernel for kernelName[%s] "
                "cmdType[%d] dataType[%s] argsType[%d] failed", aivKernelInfo.kernelName, cmdType,
                GetDataTypeEnumStr(aivKernelInfo.dataType).c_str(), aivKernelInfo.argsType), HCCL_E_RUNTIME);
        }
    }

    g_init = true;

    return HCCL_SUCCESS;
}

HcclResult UnRegisterAivKernel()
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        ACLCHECK(aclrtBinaryUnLoad(g_binHandle));
        g_aivFuncMap.clear();

        g_init = false;
    }

    return HCCL_SUCCESS;
}

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, void* args, u32 argsSize)
{
    HCCL_INFO("[ExecuteKernelLaunchInner] sendbuff [%p] recvbuff [%p] rank [%u] rankSize [%u] count [%llu] "
        "dataType [%d] reduceOp [%d] root [%u] tag [%u] isOpBase [%d] "
        "extraArgsPtr [%p] argsSize [%u]", opArgs.input,
        opArgs.output, opArgs.rank, opArgs.rankSize, opArgs.count,
        opArgs.dataType, opArgs.op, opArgs.root,
        opArgs.sliceId, opArgs.isOpBase, args, argsSize);

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr[AIV_ATTRNUM_THREE];
    attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr[0].value.schemMode = 1;
    attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr[1].value.timeoutUs.timeoutLow = GetAivTimeout();
    attr[1].value.timeoutUs.timeoutHigh = 0;
    attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    cfg.numAttrs = AIV_ATTRNUM_THREE;
    cfg.attrs = attr;

    HCCL_INFO("[ExecuteKernelLaunchInner] KernelAttr attr[0]: id=%u, schemMode=%u; attr[1]: id=%u, timeoutLow=%u, "
        "timeoutHigh=%u; attr[2]: id=%u, engineType=%u; cfg: numAttrs=%u",
        attr[0].id, attr[0].value.schemMode, attr[1].id, attr[1].value.timeoutUs.timeoutLow,
        attr[1].value.timeoutUs.timeoutHigh, attr[2].id, attr[2].value.engineType, cfg.numAttrs);

    s8* funcKey = GetFuncKey(opArgs.cmdType, opArgs.dataType, opArgs.argsType);
    aclrtFuncHandle funcHandle;
    HcclResult ret = GetKernelFunc(funcHandle, funcKey);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] errNo[0x%016llx] GetKernelFunc failed, "
        "return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret), HCCL_E_RUNTIME);

    aclError aclRet = aclrtLaunchKernelWithHostArgs(funcHandle, opArgs.numBlocks, opArgs.stream,
        &cfg, args, argsSize, nullptr, 0);
    if (aclRet == ACL_ERROR_RT_INVALID_HANDLE) {
        HCCL_WARNING("[ExecuteKernelLaunchInner] handle invalid, retry to get function");
        if (g_aivNameMap.find(funcKey) != g_aivNameMap.end()) {
            const std::string& kernelName = g_aivNameMap[funcKey];
            aclRet = aclrtBinaryGetFunction(g_binHandle, kernelName.c_str(), &funcHandle);
            CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] retry get function failed, error[%d]", aclRet), HCCL_E_RUNTIME);
            g_aivFuncMap[funcKey] = funcHandle;
            aclRet = aclrtLaunchKernelWithHostArgs(funcHandle, opArgs.numBlocks, opArgs.stream,
                &cfg, args, argsSize, nullptr, 0);
        }
    }
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner]errNo[0x%016llx] aclrtLaunchKernelWithHostArgs error[%d].",
        HCCL_ERROR_CODE(HCCL_E_RUNTIME), aclRet), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs)
{
    // Recording Logic
    if (g_recordingQueue) {
        AivInstruction ins;
        ins.opArgs = opArgs;
        // Calculate offsets
        // Note: opArgs.input is absolute address.
        if (opArgs.input >= g_baseInputAddr) {
             ins.inputOffset = opArgs.input - g_baseInputAddr;
        } else {
             ins.inputOffset = 0;
        }
        
        if (opArgs.output >= g_baseOutputAddr) {
             ins.outputOffset = opArgs.output - g_baseOutputAddr;
        } else {
             ins.outputOffset = 0;
        }
        g_recordingQueue->push_back(ins);
    }

    AivExtraKernelArgs aivExtraKernelArgs {
        opArgs.buffersIn, opArgs.input, opArgs.output,
        opArgs.rank, opArgs.rankSize, opArgs.xRankSize, opArgs.yRankSize, opArgs.zRankSize, opArgs.count, opArgs.dataType, opArgs.op, opArgs.root, opArgs.sliceId,
        opArgs.inputSliceStride, opArgs.outputSliceStride, opArgs.repeatNum, opArgs.inputRepeatStride, opArgs.outputRepeatStride,
        opArgs.isOpBase,
        reinterpret_cast<void*>(opArgs.counter.headCountMem),
        reinterpret_cast<void*>(opArgs.counter.tailCountMem), reinterpret_cast<void*>(opArgs.counter.addOneMem),
        opArgs.counter.memSize, &opArgs.extraArgs
    };
    CHK_RET(ExecuteKernelLaunchInner(opArgs, &aivExtraKernelArgs, sizeof(aivExtraKernelArgs)));
    return HCCL_SUCCESS;
}

}   // ~~ namespace hccl