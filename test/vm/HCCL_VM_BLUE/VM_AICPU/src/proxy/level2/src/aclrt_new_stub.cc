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
#include <atomic>
#include <iostream>
#include <securec.h>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "acl_rt_impl.h"
#include "dtype_common.h"
#include "rts_device.h"
#include "hccp_common.h"
#include "ascend_hal.h"
#include "hccl_vm_log.h"
#include "rts_snapshot.h"
#include "atrace_types.h"
#include "mem.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
HcclResult hrtGetDeviceType(DevType &devType)
{
    devType = DevType::DEV_TYPE_910_95;
    HCCL_VM_TRACE("[{}] Return DEV_TYPE_910_95", __func__);
    return HCCL_SUCCESS;
}

rtError_t rtOpenNetService(const rtNetServiceOpenArgs *args)
{
    // hccpThreadStatus = 1;
    return ACL_SUCCESS;
}


drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value)
{

    if (moduleType == (int32_t)MODULE_TYPE_SYSTEM && infoType == (int32_t)INFO_TYPE_VERSION) {
        int64_t hardwareVersion = 0xf00;
    } else if ((moduleType == (int32_t)MODULE_TYPE_SYSTEM) && (infoType == (int32_t)INFO_TYPE_CORE_NUM)) {
        *value = 1;
    } else if ((moduleType == (int32_t)MODULE_TYPE_AICPU)) {
        *value = 0x500;
    } else {
        *value = 0;
    }
    HCCL_VM_DEBUG("[{}] Stub devId {:d} type {:d} info{:d} value{:d}", __func__, (int)devId, (int)moduleType, (int)infoType, (int)*value);
    // *value = hardwareVersion;
    return DRV_ERROR_NONE;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    return RT_ERROR_NONE; 
}

drvError_t halGetSocVersion(uint32_t devId, char *soc_version, uint32_t len)
{
    strcpy(soc_version, "Ascend950");
    return DRV_ERROR_NONE;
}

int ra_ctx_deinit(void *ctx_handle)
{
    return 0;
}

int ra_ctx_qp_create(void *ctx_handle, struct qp_create_attr *attr, struct qp_create_info *info, void **qp_handle)
{
    return 0;
}

int ra_ctx_qp_bind(void *qp_handle, void *rem_qp_handle)
{
    return 0;
}

int ra_ctx_qp_unimport(void *ctx_handle, void *rem_qp_handle)
{
    return 0;
}

int ra_ctx_token_id_alloc(void *ctx_handle, struct hccp_token_id *info, void **token_id_handle)
{
    return 0;
}

int ra_get_sec_random(struct RaInfo *info, uint32_t *value)
{
    return 0;
}

int ra_ctx_lmem_unregister(void *ctx_handle, void *lmem_handle)
{
    return 0;
}

int ra_ctx_lmem_register(void *ctx_handle, struct mr_reg_info_t *lmem_info, void **lmem_handle)
{
    return 0;
}

int ra_ctx_rmem_unimport(void *ctx_handle, void *rmem_handle)
{
    return 0;
}

int ra_ctx_qp_import(void *ctx_handle, struct qp_import_info_t *qp_info, void **rem_qp_handle)
{
    return 0;
}

int ra_ctx_qp_unimport_async(void *rem_qp_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_ctx_lmem_unregister_async(void *ctx_handle, void *lmem_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_ctx_qp_destroy_async(void *qp_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_ctx_qp_destroy(void *qp_handle)
{
    return 0;
}

int ra_get_async_req_result(void *req_handle, int *req_result)
{
    return 0;
}

int ra_ctx_token_id_free(void *ctx_handle, void *token_id_handle)
{
    return 0;
}

int ra_ctx_qp_import_async(void *ctx_handle, struct qp_import_info_t *info, void **rem_qp_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

// int ra_ctx_lmem_register_async(void *ctx_handle, struct mr_reg_info_t *lmem_info, void **lmem_handle,
//     void **req_handle)
// {
//     *lmem_handle = reinterpret_cast<void *>(lmem_info->in.mem.addr);
//     *req_handle = reinterpret_cast<void *>(0x12345678);
//     return 0;
// }

int ra_get_tp_info_list_async(void *ctx_handle, struct get_tp_cfg *cfg, struct tp_info info_list[],
    unsigned int *num, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    *num = 1;
    return 0;
}

int ra_get_qp_context(void* qpHandle, void** qp, void** sendCq, void** recvCq)
{
    return 0;
}

int ra_get_tsqp_depth(void *rdev_handle, unsigned int *temp_depth, unsigned int *qp_num)
{
    *temp_depth = 1;
    *qp_num = 1;
    return 0;
}

int ra_set_tsqp_depth(void *rdev_handle, unsigned int temp_depth, unsigned int *qp_num)
{
    return 0;
}

int ra_get_notify_mr_info(void* handle, struct mr_info *mrInfo)
{
    return 0;
}

int ra_send_wrlist_ext(void *qp_handle, struct send_wrlist_data_ext wr[], struct send_wr_rsp op_rsp[],
    unsigned int send_num, unsigned int *complete_num)
{
    return 0;
}

int ra_register_mr(const void* handle, struct mr_info *mrInfo, void **mrHandle)
{
    *mrHandle = (void *)0xabcd;
    return ((handle == NULL) || (mrInfo == NULL)) ? -1 :0;
}

int ra_deregister_mr(const void* handle, void *mrHandle)
{
    return ((handle == NULL) || (mrHandle == NULL)) ? -1 :0;
}

int ra_is_first_used(int ins_id)
{
    return 0;
}

int ra_is_last_used(int ins_id)
{
    return 0;
}

int ra_epoll_ctl_add(const void *fd_handle, RaEpollEvent event)
{
    return 0;
}

int ra_epoll_ctl_mod(const void *fd_handle, RaEpollEvent event)
{
    return 0;
}

int ra_epoll_ctl_del(const void *fd_handle)
{
    return 0;
}


int ra_ctx_cq_create(void *ctx_handle, struct cq_info_t *info, void **cq_handle)
{
    return 0;
}

int ra_ctx_cq_destroy(void *ctx_handle, void *cq_handle)
{
    return 0;
}

int ra_ctx_update_ci(void *qp_handle, uint16_t ci)
{
    return 0;
}

int ra_get_eid_by_ip_async(void *ctx_handle, struct IpInfo ip[], union hccp_eid eid[],
    unsigned int *num, void **req_handle)
{
    return 0;
}

rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void *devInfo)
{
    return RT_ERROR_NONE;
}

int ra_ctx_qp_create_async(void *ctx_handle, struct qp_create_attr *attr, struct qp_create_info *info, void **qp_handle,
    void **req_handle)
{
    *qp_handle = reinterpret_cast<void *>(0x24681012);
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

TraStatus AtraceSubmit(TraHandle handle, const void *buffer, uint32_t bufSize)
{
    if (handle == 0) {
    return 0;
    } else if (handle == 1) {
        return -1;
    }
    return 0;
}

void AtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

int32_t AtraceCreateWithAttr(int32_t tracerType, const char *objName, const TraceAttr *attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}



TraStatus AtraceSetGlobalAttr(const TraceGlobalAttr *attr)
{
    return 0;
}

int32_t AtraceSave(TracerType tracerType, bool syncFlag)
{
    (void)(tracerType);
    (void)(syncFlag);
    return 0;
}

rtError_t rtPointerGetAttributes(rtPointerAttributes_t *attributes, const void *ptr)
{
    return 0;
}

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t *cqId, uint32_t *logicCqId)
{
    static uint32_t i = 0U;
    *logicCqId = i++;
    return 0;
}

rtError_t rtCntNotifyCreateServer(rtCntNotify_t * const cntNotify, uint64_t flags)
{
    return 0;
}

rtError_t rtsCntNotifyGetId(rtCntNotify_t cntNotify, uint32_t *notifyId)
{
    return ACL_SUCCESS;
}

rtError_t rtGetDevResAddress(rtDevResInfo *const resInfo, rtDevResAddrInfo *const addrInfo)
{
    if (resInfo == nullptr || (resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_NOTIFY_RECORD && resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_CNT_NOTIFY_BIT_WR)) {
        // 非NotifyRecord场景暂不处理
        return RT_ERROR_NONE;
    }
    // std::cout<<"[ERROR][rtGetDevResAddress] Not support...."<<std::endl;

    uint32_t notifyId = resInfo->resId;
    uint32_t len = 8;
    *(addrInfo->resAddress) = static_cast<uint64_t>(notifyId);
    *(addrInfo->len) = len;
    return RT_ERROR_NONE;
}

rtError_t rtProfRegisterCtrlCallback(uint32_t moduleId, rtProfCtrlHandle callback)
{
    return ACL_SUCCESS;
}

rtError_t rtGetSocVersion(char_t *ver, const uint32_t maxLen)
{
    return halGetSocVersion(0, ver, maxLen);
}

aclError aclrtSnapShotCallbackRegisterImpl(aclrtSnapShotStage stage, aclrtSnapShotCallBack callback, void* args)
{
    HCCL_VM_DEBUG("[{}] Stub is empty", __func__);
    return RT_ERROR_NONE;
}

aclError aclrtSnapShotCallbackUnregisterImpl(aclrtSnapShotStage stage, aclrtSnapShotCallBack callback)
{
    HCCL_VM_DEBUG("[{}] Stub is empty", __func__);
    return RT_ERROR_NONE;
}

rtError_t rtSnapShotCallbackRegister(rtSnapShotStage stage, rtSnapShotCallBack callback, void* args)
{
    aclrtSnapShotCallbackRegisterImpl((aclrtSnapShotStage)stage, (aclrtSnapShotCallBack)callback, args);
    HCCL_VM_DEBUG("[{}] Stub is empty", __func__);
    return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif  // __cplusplus