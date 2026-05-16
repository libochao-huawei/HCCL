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

thread_local bool g_hcclErrToWarn = false;

constexpr int32_t HCCL_LOG_LEVEL_INVALID = -1;
static int32_t g_hcclLogLevel = HCCL_LOG_LEVEL_INVALID;

bool HcclCheckLogLevel(int logType, int moduleId)
{
    if (UNLIKELY(g_hcclLogLevel == HCCL_LOG_LEVEL_INVALID)) {
        int32_t enableEvent = -1;
        g_hcclLogLevel = dlog_getlevel(moduleId, &enableEvent);
    }
    return (logType >= g_hcclLogLevel);
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