/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdio>
#include <map>
#include <string>
#include <mutex>
#include <securec.h>
#include "hccl_vm_log.h"
#include "hccl_sim_shm_manager.h"

constexpr size_t LOG_MSG_BUFFER_SIZE = 1024;

std::once_flag log_initialized_flag;

#ifdef __cplusplus
extern "C" {
#endif

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    return static_cast<int32_t>(true);
}

void DlogPrintStub(int level, const char *msgBuffer)
{
    std::call_once(log_initialized_flag, []() {
        LogConfig config;
        auto* proxyConfig = SHMManager::GetProxyConfig();
        if (proxyConfig != nullptr) {
            config.consoleLevel = proxyConfig->consoleLogLevel;
            config.fileLevel = proxyConfig->fileLogLevel;
        }
        config.filePath = "logs/proxy";
        InitLogger(config);
    });

    switch (level) {
    case 0:
        HCCL_VM_DEBUG(msgBuffer);
        break;
    case 1:
        HCCL_VM_INFO(msgBuffer);
        break;
    case 2:
        HCCL_VM_WARN(msgBuffer);
        break;
    case 3:
        HCCL_VM_ERROR(msgBuffer);
        break;
    default:
        HCCL_VM_TRACE(msgBuffer);
    }
}

void DlogRecord(int moduleId, int level, const char *fmt, ...)
{
    // 定义一个缓冲区，用于存储格式化后的字符串
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args); 
    va_end(args);

    DlogPrintStub(level, buffer);
}

void _Z10DlogRecordiiPKcz(int moduleId, int level, const char *fmt, ...)
{
    // 定义一个缓冲区，用于存储格式化后的字符串
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args); 
    va_end(args);

    DlogPrintStub(level, buffer);
}

#ifdef __cplusplus
}
#endif
