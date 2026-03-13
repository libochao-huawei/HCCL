/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: adaper rts v2 header file
 * Author:
 * Create: 2025-05-29
 */

#ifndef HCCLV2_ORION_ADAPTER_RTS_H
#define HCCLV2_ORION_ADAPTER_RTS_H

#include <string>
#include <unordered_map>
#include "types.h"
#include "dev_type.h"
#include "rt_external.h"

namespace Hccl {

DevType HrtGetDeviceType();
void HrtGetSocVer(char_t *chipVer, const u32 size);
void HrtUbDevQueryInfo(rtUbDevQueryCmd cmd, void *devInfo);
u32 HrtGetDevicePhyIdByIndex(u32 deviceLogicId);
void *HrtMalloc(u64 size, rtMemType_t memType);
void HrtFree(void *devPtr);
void HrtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind);
s32 HrtGetDevice();

class HcclMainboardId;
HcclResult HrtGetMainboardId(uint32_t deviceLogicId, HcclMainboardId &hcclMainboardId);

} // namespace Hccl

#endif // HCCLV2_ORION_ADAPTER_RTS_H