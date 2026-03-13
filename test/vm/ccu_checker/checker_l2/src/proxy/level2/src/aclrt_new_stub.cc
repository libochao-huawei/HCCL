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
#include "dtype_common.h"
#include "rts_device.h"
#include "hccp_common.h"
#include "ascend_hal.h"
#include "hccl_vm_log.h"
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

int RaInit(struct RaInitConfig *config)
{
    return 0;
}

int RaDeinit(struct RaInitConfig *config)
{
    return 0;
}

int RaTlvInit(struct TlvInitInfo *initInfo, unsigned int *bufferSize, void **tlvHandle)
{
    *tlvHandle = (void*)0x123456;
    return 0;
}

int RaTlvDeinit(void *tlvHandle)
{
    return 0;
}

int RaTlvRequest(void *tlvHandle, unsigned int moduleType, struct TlvMsg *sendMsg, struct TlvMsg *recvMsg)
{
    return 0;
}

drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    
    if (moduleType == (int32_t)MODULE_TYPE_SYSTEM && infoType == (int32_t)INFO_TYPE_VERSION) {
        int64_t hardwareVersion = 0xf00;
    } else if ((moduleType == (int32_t)MODULE_TYPE_SYSTEM) && (infoType == (int32_t)INFO_TYPE_CORE_NUM)) {
        *value = 1;
    } else {
        *value = 0;
    }
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

int RaCtxQpCreate(void *ctxHandle, struct QpCreateAttr *attr, struct QpCreateInfo *info, void **qpHandle)
{
    return 0;
}

int RaCtxQpBind(void *qpHandle, void *remQpHandle)
{
    return 0;
}

int RaCtxQpUnimport(void *ctxHandle, void *remQpHandle)
{
    return 0;
}

int RaCtxTokenIdAlloc(void *ctxHandle, struct HccpTokenId *info, void **tokenIdHandle)
{
    return 0;
}

int RaGetSecRandom(struct RaInfo *info, uint32_t *value)
{
    return 0;
}

int RaCtxLmemUnregister(void *ctxHandle, void *lmemHandle)
{
    return 0;
}

int RaCtxRmemUnimport(void *ctxHandle, void *rmemHandle)
{
    return 0;
}

int RaCtxQpImport(void *ctxHandle, struct QpImportInfoT *qpInfo, void **remQpHandle)
{
    return 0;
}

int RaCtxQpUnimportAsync(void *remQpHandle, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int RaCtxLmemUnregisterAsync(void *ctxHandle, void *lmemHandle, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int RaCtxQpDestroyAsync(void *qpHandle, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int RaCtxQpDestroyBatchAsync(void *ctxHandle, void *qpHandle[], unsigned int *num, void **reqHandle)
{
    HCCL_VM_ERROR("[RaCtxQpDestroyBatchAsync] Not support yet.");
    return -1;
}

int RaCtxRmemImport(void *ctxHandle, struct MrImportInfoT *rmemInfo, void **rmemHandle)
{
    HCCL_VM_ERROR("[RaCtxRmemImport] Not support yet.");
    return -1;
}

int RaCtxChanCreate(void *ctxHandle, struct ChanInfoT *chanInfo, void **chanHandle)
{
    HCCL_VM_ERROR("[RaCtxChanCreate] Not support yet.");
    return -1;
}

int RaCtxChanDestroy(void *ctxHandle, void *chanHandle)
{
    HCCL_VM_ERROR("[RaCtxChanDestroy] Not support yet.");
    return -1;
}

int RaCtxQpDestroy(void *qpHandle)
{
    return 0;
}

int ra_get_async_req_result(void *req_handle, int *req_result)
{
    return 0;
}

int RaCtxTokenIdFree(void *ctxHandle, void *tokenIdHandle)
{
    return 0;
}

int RaCtxLmemRegister(void *ctxHandle, struct MrRegInfoT *lmemInfo, void **lmemHandle)
{
    return 0;
}

int RaCtxQpImportAsync(void *ctxHandle, struct QpImportInfoT *info, void **remQpHandle, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int RaCtxLmemRegisterAsync(void *ctxHandle, struct MrRegInfoT *lmemInfo,
    void **lmemHandle, void **reqHandle)
{
    HCCL_VM_ERROR("[RaCtxLmemRegisterAsync] Not support yet");
    return -1;
}

int RaGetTpInfoListAsync(void *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    *num = 1;
    return 0;
}

int RaGetEidByIpAsync(void *ctxHandle, struct IpInfo ip[], union HccpEid eid[],
    unsigned int *num, void **reqHandle)
{
    HCCL_VM_ERROR("[RaGetEidByIpAsync] Not support yet");
    return -1;
}

int RaGetTpAttrAsync(void *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap,
    struct TpAttr *attr, void **reqHandle)
{
    HCCL_VM_ERROR("[RaGetTpAttrAsync] Not support yet");
    return -1;
}

int RaCtxQpQueryBatch(void *qpHandle[], struct JettyAttr attr[], unsigned int *num)
{
    HCCL_VM_ERROR("[RaCtxQpQueryBatch] Not support yet");
    return -1;
}

int RaCtxQpUnbind(void *qpHandle)
{
    HCCL_VM_ERROR("[RaCtxQpUnbind] Not support yet");
    return -1;
}

int RaBatchSendWr(
    void *qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[], unsigned int num, unsigned int *completeNum)
{
    HCCL_VM_ERROR("[RaBatchSendWr] Not support yet");
    return -1;
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

int RaSocketGetVnicIpInfos(
    unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num, struct IpInfo infos[])
{
    return 0;
}

int RaRdevGetSupportLite(void *rdmaHandle, int *supportLite)
{
    return 0;
}

int RaCqCreate(void *rdevHandle, struct CqAttr *attr)
{
    return 0;
}

int RaCqDestroy(void *rdevHandle, struct CqAttr *attr)
{
    return 0;
}

int RaNormalQpCreate(void *rdevHandle, struct ibv_qp_init_attr *qpInitAttr, void **qpHandle, void **qp)
{
    return 0;
}

int RaNormalQpDestroy(void *qpHandle)
{
    if(qpHandle == nullptr)
    {
        return HCCL_E_PTR;
    }
    return 0;
}

int RaSetQpAttrQos(void *qpHandle, struct QosAttr *attr)
{
    return 0;
}

int RaSetQpAttrTimeout(void *qpHandle, unsigned int *timeout)
{
    return 0;
}

int RaSetQpAttrRetryCnt(void *qpHandle, unsigned int *retryCnt)
{
    return 0;
}

int RaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info)
{
    return 0;
}

int RaCreateSrq(const void *rdmaHandle, struct SrqAttr *attr)
{
    return 0;
}

int RaDestroySrq(const void *rdmaHandle, struct SrqAttr *attr)
{
    return 0;
}

int RaCreateEventHandle(int *eventHandle)
{
    return 0;
}

int RaCtlEventHandle(int eventHandle, const void *fdHandle, int opcode,
    enum RaEpollEvent event)
{
    return 0;
}

int RaWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout,
    unsigned int maxevents, unsigned int *eventsNum)
{
    return 0;
}

int RaDestroyEventHandle(int *eventHandle)
{
    return 0;
}

int RaCreateCompChannel(const void *rdmaHandle, void **compChannel)
{
    *compChannel = (void *)0xabcd;
    return ((rdmaHandle == NULL) || (compChannel == NULL)) ? -1 :0;
}

int RaDestroyCompChannel(const void *rdmaHandle, void *compChannel)
{
    return ((rdmaHandle == NULL) || (compChannel == NULL)) ? -1 :0;
}

int RaQpCreateWithAttrs(void *rdevHandle, struct QpExtAttrs *extAttrs, void **qpHandle)
{
    return 0;
}

int RaAiQpCreate(void *rdevHandle, struct QpExtAttrs *attrs, struct AiQpInfo *info,
    void **qpHandle)
{
    return 0;
}

int RaSendWrV2(void *qpHandle, struct SendWrV2 *wr, struct SendWrRsp *opRsp)
{
    return 0;
}

int RaPollCq(void *qpHandle, bool isSendCq, unsigned int numEntries, void *wc)
{
    return 0;
}

int RaRecvWrlist(void *qpHandle, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum)
{
    return 0;
}

int RaQpBatchModify(void *rdmaHandle, void *qpHandle[], unsigned int num, int expectStatus)
{
    return 0;
}

int RaRdevGetCqeErrInfoList(void *rdmaHandle, struct CqeErrInfo *infoList,
    unsigned int *num)
{
    return 0;
}

int RaRdevInitWithBackup(struct RdevInitInfo *initInfo, struct rdev *rdevInfo,
    struct rdev *backupRdevInfo, void **rdmaHandle)
{
    return 0;
}

int RaTypicalQpCreate(void *rdevHandle, int flag, int qpMode, struct TypicalQp *qpInfo,
    void **qpHandle)
{
    return 0;
}

int RaTypicalQpModify(void *qpHandle, struct TypicalQp *localQpInfo,
    struct TypicalQp *remoteQpInfo)
{
    return 0;
}

int RaTypicalSendWr(void *qpHandle, struct SendWr *wr, struct SendWrRsp *opRsp)
{
    return 0;
}

int RaRdevGetPortStatus(void *rdmaHandle, enum PortStatus *status)
{
    return 0;
}

int RaGetQpAttr(void *qpHandle, struct QpAttr *attr)
{
    return 0;
}

int RaCtxCqCreate(void *ctxHandle, struct CqInfoT *info, void **cqHandle)
{
    return 0;
}

int RaCtxCqDestroy(void *ctxHandle, void *cqHandle)
{
    return 0;
}

int RaCtxUpdateCi(void *qpHandle, uint16_t ci)
{
    return 0;
}

rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void *devInfo)
{
    return RT_ERROR_NONE;
}

int RaSocketWhiteListAdd(void *socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    return 0;
}

int RaCtxQpCreateAsync(
    void *ctxHandle, struct QpCreateAttr *attr, struct QpCreateInfo *info, void **qpHandle, void **reqHandle)
{
    *qpHandle = reinterpret_cast<void *>(0x24681012);
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int RaSetTpAttrAsync(void *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap,
    struct TpAttr *attr, void **reqHandle)
{
    HCCL_VM_ERROR("[RaSetTpAttrAsync] Not support yet");
    return -1;
}

int RaCtxGetAuxInfo(void *ctxHandle, struct HccpAuxInfoIn *in, struct HccpAuxInfoOut *out)
{
    HCCL_VM_ERROR("[RaCtxGetAuxInfo] Not support yet");
    return -1;
}

int RaCtxGetCrErrInfoList(void *ctxHandle, struct CrErrInfo *infoList, unsigned int *num)
{
    HCCL_VM_ERROR("[RaCtxGetCrErrInfoList] Not support yet");
    return -1;
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

#ifdef __cplusplus
}
#endif  // __cplusplus