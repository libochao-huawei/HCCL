#include <unistd.h>
#include <vector>
#include <atomic>
#include <iostream>
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>

#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "runtime/base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "hccl_sim_shm_manager.h"
#include "task_status_cache.h"

#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "hccp_common.h"
#include "ip_address.h"
#include "sim_runner_common.h"
#include "hccl_vm_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//////////////////////RDMA/////////////////////////////

int RaSocketGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
                           struct IpInfo infos[])
{
    infos[0].family = AF_INET;
    infos[0].ip.addr.s_addr = inet_addr("127.0.0.1");
    HCCL_VM_INFO("[HCCP] [{}] stub addr:127.0.0.1", __func__);
    return 0;
}

int RaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int *interfaceVersion)
{
    HCCL_VM_INFO("[HCCP] [{}] stub set *interfaceVersion:1", __func__);
    *interfaceVersion = 1;
    return 0;
}

int RaGetTsqpDepth(void *rdevHandle, unsigned int *tempDepth, unsigned int *qpNum)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}
int RaSetTsqpDepth(void *rdevHandle, unsigned int tempDepth, unsigned int *qpNum)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRdevGetSupportLite(void *rdmaHandle, int *supportLite)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSocketSetWhiteListStatus(unsigned int enable)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSocketGetWhiteListStatus(unsigned int *enable)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaNormalQpCreate(void *rdevHandle, struct ibv_qp_init_attr *qpInitAttr, void **qpHandle, void **qp)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaNormalQpDestroy(void *qpHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    if (qpHandle == nullptr) {
        return HCCL_E_PTR;
    }
    return 0;
}

int RaMrReg(void *qpHandle, struct MrInfoT *info)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaMrDereg(void *qpHandle, struct MrInfoT *info)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRegisterMr(const void *rdmaHandle, struct MrInfoT *info, void **mrHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRemapMr(const void *rdmaHandle, struct MemRemapInfo info[], unsigned int num)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaDeregisterMr(const void *rdmaHandle, void *mrHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSendWr(void *qpHandle, struct SendWr *wr, struct SendWrRsp *opRsp)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSetQpAttrQos(void *qpHandle, struct QosAttr *attr)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSetQpAttrTimeout(void *qpHandle, unsigned int *timeout)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSetQpAttrRetryCnt(void *qpHandle, unsigned int *retryCnt)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaCreateSrq(const void *rdmaHandle, struct SrqAttr *attr)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaDestroySrq(const void *rdmaHandle, struct SrqAttr *attr)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaCreateEventHandle(int *eventHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaCtlEventHandle(int eventHandle, const void *fdHandle, int opcode, enum RaEpollEvent event)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout, unsigned int maxevents,
                      unsigned int *eventsNum)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaDestroyEventHandle(int *eventHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaCreateCompChannel(const void *rdmaHandle, void **compChannel)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    *compChannel = (void *)0xabcd;
    return ((rdmaHandle == NULL) || (compChannel == NULL)) ? -1 : 0;
}

int RaDestroyCompChannel(const void *rdmaHandle, void *compChannel)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return ((rdmaHandle == NULL) || (compChannel == NULL)) ? -1 : 0;
}

int RaLoopbackQpCreate(void *rdevHandle, struct LoopbackQpPair *qpPair, void **qpHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaQpConnectAsync(void *qpHandle, const void *fdHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSendWrV2(void *qpHandle, struct SendWrV2 *wr, struct SendWrRsp *opRsp)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPollCq(void *qpHandle, bool isSendCq, unsigned int numEntries, void *wc)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRecvWrlist(void *qpHandle, struct RecvWrlistData *wr, unsigned int recvNum, unsigned int *completeNum)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetQpContext(void *qpHandle, void **qp, void **sendCq, void **recvCq)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaQpBatchModify(void *rdmaHandle, void *qpHandle[], unsigned int num, int expectStatus)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRdevGetCqeErrInfoList(void *rdmaHandle, struct CqeErrInfo *infoList, unsigned int *num)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRdevGetHandle(unsigned int phyId, void **rdmaHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSaveSnapshot(struct RaInfo *info, enum SaveSnapshotAction action)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRestoreSnapshot(struct RaInfo *info)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}
int RaRdevInitWithBackup(struct RdevInitInfo *initInfo, struct rdev *rdevInfo, struct rdev *backupRdevInfo,
                         void **rdmaHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetQpStatus(void *qpHandle, int *status)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSendWrlist(void *qpHandle, struct SendWrlistData wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
                 unsigned int *completeNum)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSendWrlistExt(void *qpHandle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
                    unsigned int *completeNum)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSendNormalWrlist(void *qpHandle, struct WrInfo wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
                       unsigned int *completeNum)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetNotifyBaseAddr(void *rdevHandle, unsigned long long *va, unsigned long long *size)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetNotifyMrInfo(void *rdevHandle, struct MrInfoT *info)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaInit(struct RaInitConfig *config)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaDeinit(struct RaInitConfig *config)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetTlsEnable(struct RaInfo *info, bool *tlsEnable)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRdevInitV2(struct RdevInitInfo initInfo, struct rdev rdevInfo, void **rdmaHandle)
{
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("[{}] can not get CurrContext: {:d}", __func__, runner.current_ctx_id);
        return 0;
    }

    sim::RaDevice dev{};
    dev.device_id = currCtx->device_id;
    auto id = RunnerDB::Add<sim::RaDevice>(dev);

    *rdmaHandle = (void *)id;
    HCCL_VM_INFO("[HCCP] [{}] add id {:d}", __func__, id);
    return 0;
}

int RaRdevInit(int mode, unsigned int notifyType, struct rdev rdevInfo, void **rdmaHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    struct RdevInitInfo initInfo;
    RaRdevInitV2(initInfo, rdevInfo, rdmaHandle);
    return 0;
}

int RaRdevDeinit(void *rdmaHandle, unsigned int notifyType)
{
    uint64_t id = (uint64_t)rdmaHandle;
    RunnerDB::Delete<sim::RaDevice>(id);
    HCCL_VM_INFO("[HCCP] [{}] delete id {:d}", __func__, id);
    return 0;
}

int RaCqCreate(void *rdevHandle, struct CqAttr *attr)
{
    uint64_t raDevId = (uint64_t)rdevHandle;
    sim::RaCQ cq{};
    cq.ra_dev_id = raDevId;
    auto id = RunnerDB::Add<sim::RaCQ>(cq);

    *(attr->qpContext) = (void *)id;

    HCCL_VM_INFO("[HCCP] [{}] stub RaDev {:d} add RaCQ id:{:d}", __func__, raDevId, id);
    return 0;
}

int RaCqDestroy(void *rdevHandle, struct CqAttr *attr)
{
    uint64_t raDevId = (uint64_t)rdevHandle;
    uint64_t cqId = (uint64_t)attr->qpContext;
    RunnerDB::Delete<sim::RaCQ>(cqId);

    HCCL_VM_INFO("[HCCP] [{}] stub RaDev {:d} delete RaCQ id:{:d}", __func__, raDevId, cqId);
    return 0;
}

int RaQpCreate(void *rdevHandle, int flag, int qpMode, void **qpHandle)
{
    uint64_t raDevId = (uint64_t)rdevHandle;
    sim::RaQP qp{};
    qp.state = 0;
    qp.ra_dev_id = raDevId;

    auto id = RunnerDB::Add<sim::RaQP>(qp);
    *qpHandle = (void *)id;

    HCCL_VM_INFO("[HCCP] [{}] RaDev {:d} create QP id {:d}", __func__, raDevId, id);
    return 0;
}

int RaQpCreateWithAttrs(void *rdevHandle, struct QpExtAttrs *extAttrs, void **qpHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    int ret = RaQpCreate(rdevHandle, 0, 0, qpHandle);
    return ret;
}

int RaAiQpCreate(void *rdevHandle, struct QpExtAttrs *attrs, struct AiQpInfo *info, void **qpHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    int ret = RaQpCreate(rdevHandle, 0, 0, qpHandle);
    return ret;
}

int RaTypicalQpCreate(void *rdevHandle, int flag, int qpMode, struct TypicalQp *qpInfo, void **qpHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);

    int ret = RaQpCreate(rdevHandle, 0, 0, qpHandle);
    return ret;
}

int RaQpDestroy(void *qpHandle)
{
    uint64_t qpId = (uint64_t)qpHandle;
    RunnerDB::Delete<sim::RaQP>(qpId);

    HCCL_VM_INFO("[HCCP] [{}] destory QP {:d}", __func__, qpId);
    return 0;
}

int RaTypicalQpModify(void *qpHandle, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaTypicalSendWr(void *qpHandle, struct SendWr *wr, struct SendWrRsp *opRsp)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaRdevGetPortStatus(void *rdmaHandle, enum PortStatus *status)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetQpAttr(void *qpHandle, struct QpAttr *attr)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSocketWhiteListAdd(void *socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaSocketWhiteListDel(void *socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}
int RaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], unsigned int num, unsigned int creditLimit)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaGetIfnum(struct RaGetIfattr *config, unsigned int *num)
{
    *num = 1;
    HCCL_VM_INFO("[HCCP] [{}] stub set num:1", __func__);
    return 0;
}

int RaGetIfaddrs(struct RaGetIfattr *config, struct InterfaceInfo interfaceInfos[], unsigned int *num)
{
    *num = 1;
    interfaceInfos[0].family = AF_INET;
    interfaceInfos[0].scopeId = 0;
    interfaceInfos[0].ifaddr.ip.addr.s_addr = inet_addr("127.0.0.1");
    HCCL_VM_INFO("[HCCP] [{}] stub addr:127.0.0.1", __func__);
    return 0;
}

int RaTlvInit(struct TlvInitInfo *initInfo, unsigned int *bufferSize, void **tlvHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    *tlvHandle = (void *)0x123456;
    return 0;
}

int RaTlvDeinit(void *tlvHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaTlvRequest(void *tlvHandle, unsigned int moduleType, struct TlvMsg *sendMsg, struct TlvMsg *recvMsg)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPingInit(struct PingInitAttr *initAttr, struct PingInitInfo *initInfo, void **pingHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPingDeinit(void *pingHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPingTargetAdd(void *pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPingTaskStart(void *pingHandle, struct PingTaskAttr *attr)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPingGetResults(void *pingHandle, struct PingTargetResult target[], uint32_t *num)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPingTargetDel(void *pingHandle, struct PingTargetCommInfo target[], uint32_t num)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}

int RaPingTaskStop(void *pingHandle)
{
    HCCL_VM_INFO("[HCCP] [{}] stub", __func__);
    return 0;
}


#ifdef __cplusplus
}
#endif  // __cplusplus