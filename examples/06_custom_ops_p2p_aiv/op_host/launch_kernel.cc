/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "launch_kernel.h"

#include <mutex>
#include <string>
#include <vector>

namespace ops_hccl_p2p_aiv {
namespace {

std::mutex g_kernelMutex;
bool g_kernelLoaded = false;
aclrtBinHandle g_binHandle = nullptr;
aclrtFuncHandle g_funcHandle = nullptr;
const char *kKernelBinaryName = "hccl_custom_p2p_aiv_kernels.o";
const char *kKernelEntryName = "HcclP2pAivKernel";

HcclResult LoadKernelBinary()
{
    std::lock_guard<std::mutex> lock(g_kernelMutex);
    if (g_kernelLoaded) {
        return HCCL_SUCCESS;
    }

    const std::vector<std::string> candidates = {
        kKernelBinaryName,
        std::string("./") + kKernelBinaryName,
        std::string("./testcase/") + kKernelBinaryName,
        std::string("./build/testcase/") + kKernelBinaryName,
        std::string("./build/op_kernel/") + kKernelBinaryName,
        std::string("../testcase/") + kKernelBinaryName,
        std::string("../op_kernel/") + kKernelBinaryName,
    };

    aclError aclRet = ACL_SUCCESS;
    std::string loadedPath;
    for (const auto &candidate : candidates) {
        aclRet = aclrtBinaryLoadFromFile(candidate.c_str(), nullptr, &g_binHandle);
        if (aclRet == ACL_SUCCESS) {
            loadedPath = candidate;
            break;
        }
    }
    if (g_binHandle == nullptr) {
        HCCL_ERROR("aclrtBinaryLoadFromFile failed for all candidates, lastRet=%d", aclRet);
        return HCCL_E_NOT_FOUND;
    }

    aclRet = aclrtBinaryGetFunction(g_binHandle, kKernelEntryName, &g_funcHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("aclrtBinaryGetFunction failed, entry=%s ret=%d", kKernelEntryName, aclRet);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("loaded kernel binary from %s", loadedPath.c_str());
    g_kernelLoaded = true;
    return HCCL_SUCCESS;
}

} // namespace

HcclResult LaunchKernel(const P2pAivKernelParam &param, aclrtStream stream)
{
    CHK_PTR_NULL(stream);
    CHK_RET(LoadKernelBinary());

    aclrtLaunchKernelAttr attrs[3] = {};
    attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[0].value.schemMode = 1;
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attrs[1].value.timeoutUs.timeoutLow = kP2pAivKernelLaunchTimeoutSec * 1000000U;
    attrs[1].value.timeoutUs.timeoutHigh = 0;
    attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attrs[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;

    aclrtLaunchKernelCfg cfg;
    cfg.attrs = attrs;
    cfg.numAttrs = 3;

    aclError aclRet = aclrtLaunchKernelWithHostArgs(
        g_funcHandle,
        param.blockNum,
        stream,
        &cfg,
        const_cast<P2pAivKernelParam *>(&param),
        sizeof(P2pAivKernelParam),
        nullptr,
        0);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("aclrtLaunchKernelWithHostArgs failed ret=%d", aclRet);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_p2p_aiv
