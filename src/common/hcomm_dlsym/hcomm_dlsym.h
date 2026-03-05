#ifndef HCOMM_DLSYM_H
#define HCOMM_DLSYM_H

#include "hccl_types.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

// 动态库管理接口（大驼峰命名）
int HcommDlInit(void);
void HcommDlFini(void);
int GetHcommVersion(void);

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