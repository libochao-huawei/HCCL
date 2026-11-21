/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file ccu_stuck_counter_trampoline.h
 * @brief CCU 注册期计数参数填充、trampoline 与注册函数选择（仅 host 使用）。
 */

#ifndef CCU_STUCK_COUNTER_TRAMPOLINE_H
#define CCU_STUCK_COUNTER_TRAMPOLINE_H

#include <cstdint>

#include "alg_param.h"
#include "ccu_stuck_counter.h"
#include "log.h"

namespace ops_hccl {

inline void FillCcuStuckCounter(
    CcuKernelArgBase& arg, bool enabled, uint64_t headAddr, uint64_t tailAddr, uint64_t addOneAddr, uint64_t token)
{
    arg.stuckCounterEnabled = enabled;
    arg.stuckHeadAddr = headAddr;
    arg.stuckTailAddr = tailAddr;
    arg.stuckAddOneAddr = addOneAddr;
    arg.stuckCounterToken = token;
}

inline CcuResult CcuStuckCounterTrampoline(CcuKernelArg arg)
{
    auto* argBase = static_cast<CcuKernelArgBase*>(arg);
    if (argBase == nullptr || argBase->stuckOrigKernelFunc == nullptr) {
        HCCL_ERROR("[StuckDetect][CcuEmit] orig kernel missing, arg[%p].", arg);
        return CCU_E_INTERNAL;
    }
    CcuStuckCounter stuckCounter(argBase);
    CcuResult headRet = stuckCounter.Head();
    if (headRet != CCU_SUCCESS) {
        HCCL_DEBUG(
            "[StuckDetect][CcuEmit] mark failed, ret[%d], dst[0x%llx], tokenGot[%d].", static_cast<int>(headRet),
            argBase->stuckHeadAddr, argBase->stuckCounterToken != 0);
    }
    CcuResult ret = reinterpret_cast<CcuKernelFunc>(argBase->stuckOrigKernelFunc)(arg);
    CcuResult tailRet = stuckCounter.Tail();
    if (tailRet != CCU_SUCCESS) {
        HCCL_DEBUG(
            "[StuckDetect][CcuEmit] mark failed, ret[%d], dst[0x%llx], tokenGot[%d].", static_cast<int>(tailRet),
            argBase->stuckTailAddr, argBase->stuckCounterToken != 0);
    }
    return ret;
}

inline void InstallCcuStuckTrampoline(CcuKernelInfo& kernelInfo)
{
    auto* argBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
    if (argBase == nullptr || !argBase->stuckCounterEnabled) {
        return;
    }
    argBase->stuckOrigKernelFunc = kernelInfo.kernelFunc;
    HCCL_DEBUG(
        "[StuckDetect][CcuSelect] kernel[%s] install trampoline, tokenGot[%d].", kernelInfo.kernelFuncName,
        argBase->stuckCounterToken != 0);
}

inline void* GetCcuRegisteredKernelFunc(const CcuKernelInfo& kernelInfo)
{
    auto* argBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
    if (argBase == nullptr || !argBase->stuckCounterEnabled || argBase->stuckOrigKernelFunc == nullptr) {
        HCCL_DEBUG("[StuckDetect][CcuSelect] kernel[%s] -> original.", kernelInfo.kernelFuncName);
        return kernelInfo.kernelFunc;
    }
    return reinterpret_cast<void*>(&CcuStuckCounterTrampoline);
}

} // namespace ops_hccl

#endif // CCU_STUCK_COUNTER_TRAMPOLINE_H
