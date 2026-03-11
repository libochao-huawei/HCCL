#ifndef HCOMM_DLSYM_H
#define HCOMM_DLSYM_H

#include "hccl_types.h"
#include "dtype_common.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

// 动态库管理接口（大驼峰命名）
int HcommDlInit(void);
void HcommDlFini(void);
int GetHcommVersion(void);

// 功能支持情况查询
bool HcommIsProfilingSupported();
bool HcommIsExportThreadSupported();

// 新增：查询函数声明
bool HcommIsSupportHcclGetRankId(void);
bool HcommIsSupportHcclGetRankSize(void);
bool HcommIsSupportHcclRankGraphGetLayers(void);
bool HcommIsSupportHcclRankGraphGetRanksByLayer(void);
bool HcommIsSupportHcclRankGraphGetRankSizeByLayer(void);
bool HcommIsSupportHcclRankGraphGetTopoTypeByLayer(void);
bool HcommIsSupportHcclRankGraphGetInstSizeListByLayer(void);
bool HcommIsSupportHcclRankGraphGetLinks(void);
bool HcommIsSupportHcclRankGraphGetTopoInstsByLayer(void);
bool HcommIsSupportHcclRankGraphGetTopoType(void);
bool HcommIsSupportHcclRankGraphGetRanksByTopoInst(void);
bool HcommIsSupportHcclGetHeterogMode(void);
bool HcommIsSupportHcclRankGraphGetEndpointNum(void);
bool HcommIsSupportHcclRankGraphGetEndpointDesc(void);
bool HcommIsSupportHcclRankGraphGetEndpointInfo(void);

bool HcommIsSupportHcclGetHcclBuffer(void);
bool HcommIsSupportHcclGetRemoteIpcHcclBuf(void);
bool HcommIsSupportHcclThreadAcquire(void);
bool HcommIsSupportHcclThreadAcquireWithStream(void);
bool HcommIsSupportHcclChannelAcquire(void);
bool HcommIsSupportHcclChannelGetHcclBuffer(void);
bool HcommIsSupportHcclEngineCtxCreate(void);
bool HcommIsSupportHcclEngineCtxGet(void);
bool HcommIsSupportHcclEngineCtxCopy(void);
bool HcommIsSupportHcclTaskRegister(void);
bool HcommIsSupportHcclTaskUnRegister(void);
bool HcommIsSupportHcclDevMemAcquire(void);
bool HcommIsSupportHcclThreadExportToCommEngine(void);
bool HcommIsSupportHcclChannelGetRemoteMems(void);
bool HcommIsSupportHcclCommMemReg(void);
bool HcommIsSupportHcclEngineCtxDestroy(void);

bool HcommIsSupportHcommLocalCopyOnThread(void);
bool HcommIsSupportHcommLocalReduceOnThread(void);
bool HcommIsSupportHcommThreadNotifyRecordOnThread(void);
bool HcommIsSupportHcommThreadNotifyWaitOnThread(void);
bool HcommIsSupportHcommAclrtNotifyRecordOnThread(void);
bool HcommIsSupportHcommAclrtNotifyWaitOnThread(void);
bool HcommIsSupportHcommWriteOnThread(void);
bool HcommIsSupportHcommWriteReduceOnThread(void);
bool HcommIsSupportHcommWriteWithNotifyOnThread(void);
bool HcommIsSupportHcommWriteReduceWithNotifyOnThread(void);
bool HcommIsSupportHcommReadOnThread(void);
bool HcommIsSupportHcommReadReduceOnThread(void);
bool HcommIsSupportHcommWriteNbi(void);
bool HcommIsSupportHcommWriteWithNotifyNbi(void);
bool HcommIsSupportHcommReadNbi(void);
bool HcommIsSupportHcommChannelNotifyRecordOnThread(void);
bool HcommIsSupportHcommChannelNotifyRecord(void);
bool HcommIsSupportHcommChannelNotifyWaitOnThread(void);
bool HcommIsSupportHcommChannelNotifyWait(void);
bool HcommIsSupportHcommBatchModeStart(void);
bool HcommIsSupportHcommBatchModeEnd(void);
bool HcommIsSupportHcommAcquireComm(void);
bool HcommIsSupportHcommReleaseComm(void);
bool HcommIsSupportHcommSymWinGetPeerPointer(void);
bool HcommIsSupportHcommThreadSynchronize(void);
bool HcommIsSupportHcommSendRequest(void);
bool HcommIsSupportHcommWaitResponse(void);
bool HcommIsSupportHcommFlush(void);
bool HcommIsSupportHcommChannelFence(void);

HcclResult __attribute__((weak)) hrtGetDeviceType(DevType &devType);

HcclResult __attribute__((weak)) HcclAllReduceInner(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclBroadcastInner(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream);

HcclResult __attribute__((weak)) HcclReduceScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclReduceScatterVInner(void *sendBuf, const void *sendCounts, const void *sendDispls,
    void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAllGatherInner(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAllGatherVInner(void *sendBuf, uint64_t sendCount, void *recvBuf,
    const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclSendInner(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                           HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclRecvInner(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                           HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAlltoAllVCInner(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                 const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAlltoAllVInner(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAlltoAllInner(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                               const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
                               HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclReduceInner(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                             HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclBatchSendRecvInner(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclCreateOpResCtxInner(HcclComm comm, uint8_t opType, HcclDataType srcDataType, HcclDataType dstDataType,
                                          HcclReduceOp reduceType, uint64_t count, char *algConfig, uint32_t commEngine, void **opResCtx);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DLSYM_H