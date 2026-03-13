/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: rts stub header
 */

#ifndef HCCL_SIM_RTS_STUB_H
#define HCCL_SIM_RTS_STUB_H

#include "runtime/rt.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
    rtError_t rtHostMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId);
    rtError_t rtHostFree(void *devPtr);
    rtError_t rtNotifyAicpuRecord(int notifyId, int notifyCnt, int streamId);
    rtError_t rtNotifyAicpuWait(int notifyId, int notifyCnt, int streamId);
    rtError_t rtMemcpyAicpuAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind, int streamId);
    rtError_t rtReduceAicpuAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtRecudeKind_t kind, rtDataType_t type, int streamId);
}
#endif

#endif