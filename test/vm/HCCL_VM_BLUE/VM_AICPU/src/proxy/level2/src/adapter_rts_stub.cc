/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <vector>
#include <map>
#include <dlfcn.h>
#include <stdio.h>
#include <atomic>
#include <iostream>
#include <securec.h>
#include "hccl_dl.h"
#include "hccp.h"
#include "hccp_async.h"
#include "hccp_ping.h"
#include "hccp_tlv.h"
#include "tsd_client.h"
#include "ascend_hal.h"
#include "atrace_pub.h"
#include "rts_device.h"
#include "ascend_hal_mdc.h"
#include "adapter_rts_common.h"
#include "hccl_vm_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if !defined(weak_alias)
#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#define _weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)));
#endif

#if !defined(strong_alias)
#define strong_alias(name, aliasname) _strong_alias(name, aliasname)
#define _strong_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((alias(#name)));
#endif

int RaIsFirstUsed(int insId)
{
    HCCL_VM_ERROR("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaIsLastUsed(int insId)
{
    HCCL_VM_ERROR("[HCCP] [{}] stub", __func__);
    return 0;
}

int ibv_get_cq_event_stub(struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cq_context)
{
    HCCL_VM_TRACE("[{}] Stub", __func__);
    if (!channel) {
        return -1;
    }
    return 0;
}

void ibv_ack_cq_events_stub(struct ibv_cq *cq, unsigned int nevents)
{
    HCCL_VM_TRACE("[{}] Stub", __func__);
    return;
}

void ibv_query_qp_stub(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr)
{
    HCCL_VM_TRACE("[{}] Stub", __func__);
    return;
}

int32_t UtraceCreateWithAttr(int32_t tracerType, const char *objName, const TraceAttr *attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}

HcclResult UtraceSubmit(int32_t handle, const void *buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    return HCCL_SUCCESS;
}

int32_t UtraceSetGlobalAttr(const TraceGlobalAttr *attr)
{
    (void)(attr);
    return 0;
}

int32_t UtraceSave(TracerType tracerType, bool syncFlag)
{
    (void)(tracerType);
    (void)(syncFlag);
    return 0;
}

void UtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

int ibv_exp_post_send_stub(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr,
                      struct ibv_post_send_ext_attr *ext_attr, struct ibv_post_send_ext_resp *ext_resp)
{
    return 0;
}

int ibv_ext_post_send_stub(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr,
                           struct ibv_post_send_ext_attr *ext_attr, struct ibv_post_send_ext_resp *ext_resp)
{
    return 0;
}

HcclResult hrtTsdCapabilityGet(uint32_t deviceLogicId, int32_t type, uint64_t ptr)
{
    printf("[STUB][hrtTsdCapabilityGet] deviceLogicId:%u type:%u ptr:%lu\n", deviceLogicId, type, ptr);
    return (HcclResult)0;
}

HcclResult hrtGetDeviceInfo(u32 deviceId, HcclRtDeviceModuleType hcclModuleType, HcclRtDeviceInfoType hcclInfoType,
                            s64 &val)
{
    printf("[STUB][hrtGetDeviceInfo] deviceId:%u hcclModuleType:%d HcclRtDeviceInfoType:%d\n", deviceId,
           (int)hcclModuleType, (int)hcclInfoType);
    return (HcclResult)0;
}

namespace hccl
{
void *__HcclDlopenSub(const char *libName, int mode)
{
    static int dlAclRtHandle;
    return &dlAclRtHandle;
}

void *__HcclDlsymSub(void *handle, const char *funcName)
{
    void *addr = dlsym(RTLD_DEFAULT, funcName);
    if (addr == nullptr) {
        HCCL_VM_ERROR("[HcclDlsymSub] not find {} Error: [{}]\n", funcName, dlerror());
        return nullptr;
    }
    // HCCL_VM_DEBUG("[HcclDlsymSub] [{}]\n", funcName);
    return addr;
}

HcclResult __hrtOpenNetServiceSub(rtNetServiceOpenArgs *openArgs)
{
    return HCCL_SUCCESS;
}

HcclResult __hrtCloseNetServiceSub()
{
    return HCCL_SUCCESS;
}

int __HcclDlcloseSub(void *handle)
{
    handle = nullptr;
    return 0;
}

strong_alias(__HcclDlopenSub, HcclDlopen);
strong_alias(__HcclDlcloseSub, HcclDlclose);
strong_alias(__HcclDlsymSub, HcclDlsym);
strong_alias(__hrtOpenNetServiceSub, hrtOpenNetService);
strong_alias(__hrtCloseNetServiceSub, hrtCloseNetService);
}

#ifdef __cplusplus
}
#endif  // __cplusplus