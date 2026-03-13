/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl mmpa stub
 */

#include "mmpa_api.h"

VOID *mmDlsym(VOID *handle, const CHAR *funcName)
{
    return nullptr;
}

CHAR *mmDlerror()
{
    static char errInfo[10] = "ErrorInfo";
    return errInfo;
}

VOID *mmDlopen(const CHAR *fileName, INT32 mode)
{
    return nullptr;
}

INT32 mmDlclose(VOID *handle)
{
    return 0;
}

INT32 mmDladdr(VOID *addr, mmDlInfo *info)
{
    return 0;
}