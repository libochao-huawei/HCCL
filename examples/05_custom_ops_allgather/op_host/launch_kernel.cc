/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "launch_kernel.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>
#include <string>
#include "acl/acl.h"

namespace ops_hccl_allgather {

// Global variables for kernel registration
static bool g_init = false;
static std::mutex g_mut;
static aclrtBinHandle g_binHandle = nullptr;
static aclrtFuncHandle g_funcHandle = nullptr;

// Constants
const std::string AIV_BINARY_NAME = "hccl_custom_allgather_kernels.o"; // Ensure this file exists or update path
const std::string KERNEL_NAME = "HcclAllGatherAivKernel";

// Helper to read binary file
static HcclResult LoadBinaryFromFile(const std::string& fileName, void*& buffer, size_t& length) {
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        HCCL_ERROR("[LoadBinaryFromFile] Failed to open file: %s", fileName.c_str());
        return HCCL_E_NOT_FOUND;
    }
    
    length = file.tellg();
    file.seekg(0, std::ios::beg);
    
    buffer = new(std::nothrow) char[length];
    if (buffer == nullptr) {
        HCCL_ERROR("[LoadBinaryFromFile] Failed to allocate memory for binary, size: %zu", length);
        return HCCL_E_INTERNAL;
    }
    
    if (!file.read(static_cast<char*>(buffer), length)) {
        HCCL_ERROR("[LoadBinaryFromFile] Failed to read file: %s", fileName.c_str());
        delete[] static_cast<char*>(buffer);
        buffer = nullptr;
        return HCCL_E_INTERNAL;
    }
    
    return HCCL_SUCCESS;
}

// Register Kernel Logic (Adapted from hccl_aiv_utils.cc)
HcclResult RegisterKernel() {
    std::lock_guard<std::mutex> guard(g_mut);
    if (g_init) {
        return HCCL_SUCCESS;
    }

    // 1. Get Binary Path (In this example, we assume it's in the current directory or a specific path)
    // You might need to adjust this logic to find the binary correctly
    std::string binPath = AIV_BINARY_NAME;
    
    // 2. Load Binary
    void* binBuffer = nullptr;
    size_t binSize = 0;
    CHK_RET(LoadBinaryFromFile(binPath, binBuffer, binSize));
    
    // 3. Register Binary with ACL
    aclError aclRet = aclrtBinaryLoad(binBuffer, binSize, &g_binHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[RegisterKernel] aclrtBinaryLoad failed, ret: %d", aclRet);
        delete[] static_cast<char*>(binBuffer);
        return HCCL_E_INTERNAL;
    }
    
    // Buffer can be freed after load? Check ACL docs. Usually yes if it copies.
    // If ACL doesn't copy, we must keep it. 
    // aclrtBinaryLoad documentation says "The memory needs to be managed by the caller".
    // Wait, usually it means we must keep it alive if it's used? 
    // But standard practice often keeps it. Let's keep it in a global vector or just leak it for now (singleton).
    // Or better, make it static.
    // To avoid memory leak in clean shutdown, we should store it.
    static std::vector<char> g_binBufferVec; 
    // Actually, let's just keep it simple. If we want to be safe, we don't delete it.
    
    // 4. Get Function Handle
    aclRet = aclrtBinaryGetFunction(g_binHandle, KERNEL_NAME.c_str(), &g_funcHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[RegisterKernel] aclrtBinaryGetFunction failed for %s, ret: %d", KERNEL_NAME.c_str(), aclRet);
        return HCCL_E_INTERNAL;
    }

    g_init = true;
    HCCL_INFO("[RegisterKernel] Kernel registered successfully: %s", KERNEL_NAME.c_str());
    return HCCL_SUCCESS;
}

// Execute Kernel Launch Logic (Adapted from hccl_aiv_utils.cc)
HcclResult ExecuteKernelLaunch(const OpParam &param, aclrtStream stream) {
    if (!g_init) {
        CHK_RET(RegisterKernel());
    }

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr[3]; // Adapting 3 attributes from utils
    
    // 1. Scheme Mode (1 = AIV)
    attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr[0].value.schemMode = 1;
    
    // 2. Timeout
    attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr[1].value.timeoutUs.timeoutLow = CUSTOM_TIMEOUT * 1000000; // CUSTOM_TIMEOUT is in seconds? common.h says 1800.
    // In common.h: constexpr uint32_t CUSTOM_TIMEOUT = 1800;
    // In utils: CUSTOM_TIMEOUT * TIME_S_TO_US.
    // Let's assume common.h CUSTOM_TIMEOUT is seconds.
    attr[1].value.timeoutUs.timeoutLow = CUSTOM_TIMEOUT * 1000000;
    attr[1].value.timeoutUs.timeoutHigh = 0;
    
    // 3. Engine Type
    attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV; // 1 = AIV
    
    cfg.numAttrs = 3;
    cfg.attrs = attr;

    // Launch with Host Args
    // We pass 'param' as the argument struct.
    // Note: The kernel must expect OpParam by value.
    aclError aclRet = aclrtLaunchKernelWithHostArgs(g_funcHandle, 1, stream, &cfg, 
                                                    const_cast<OpParam*>(&param), sizeof(OpParam), 
                                                    nullptr, 0);
    
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[ExecuteKernelLaunch] aclrtLaunchKernelWithHostArgs failed, ret: %d", aclRet);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(OpParam &param, aclrtStream stream) {
    return ExecuteKernelLaunch(param, stream);
}

}

