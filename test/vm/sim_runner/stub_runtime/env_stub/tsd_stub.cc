/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: tsd stub
 */

#include "tsd_stub.h"

#ifdef __cplusplus
extern "C" {
uint32_t TsdOpen(const uint32_t logicDeviceId, const uint32_t rankSize)
{
    return 0;
}

uint32_t TsdCapabilityGet(uint32_t deviceLogicId, int32_t type, uint64_t ptr)
{
    return 0;
}

uint32_t TsdProcessOpen(const uint32_t logicDeviceId, ProcOpenArgs *openArgs)
{
    return 0;
}

uint32_t ProcessCloseSubProcList(const uint32_t logicDeviceId, const ProcStatusParam *closeList,
    const uint32_t listSize)
{
    return 0;
}

uint32_t TsdClose(const uint32_t phyDeviceId)
{
    return 0;
}
}
#endif