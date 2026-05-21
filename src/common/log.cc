/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log.h"

#include <unordered_map>

thread_local bool g_hcclErrToWarn = false;

static thread_local std::unordered_map<int32_t, int32_t> g_logLevelCache;

static int32_t GetLogLevel(int32_t moduleId)
{
    auto iter = g_logLevelCache.find(moduleId);
    if (iter != g_logLevelCache.end()) {
        return iter->second;
    }

    int32_t enableEvent = -1;
    int32_t logLevel = dlog_getlevel(moduleId, &enableEvent);
    g_logLevelCache.emplace(moduleId, logLevel);
    return logLevel;
}

bool HcclCheckLogLevel(int logType, int moduleId)
{
    return (logType >= GetLogLevel(moduleId));
}

void SetErrToWarnSwitch(bool flag)
{
    if (g_hcclErrToWarn != flag) {
        g_hcclErrToWarn = flag;
    }
}

bool IsErrorToWarn()
{
    return g_hcclErrToWarn;
}
