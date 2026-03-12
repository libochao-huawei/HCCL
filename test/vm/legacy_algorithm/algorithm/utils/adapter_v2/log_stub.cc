/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:日志模块
 * Author: zhujiale
 * Create: 2023-12-12
 */
#include <slog.h>
#include <slog_api.h>
#include <stdint.h>
#include <syslog.h>
#include <stdio.h>

bool HcclCheckLogLevel(int logType, int moduleId)
{
    return (CheckLogLevel(moduleId, logType) == 1) ? true : false;
}

bool IsErrorToWarn()
{
    return false;
}

bool IsRunInfoLogPrintToScreen()
{
    return false;
}