#ifndef HCCL_CUSTOM_DOUBLE_ALLGATHER_H
#define HCCL_CUSTOM_DOUBLE_ALLGATHER_H

#include "hccl_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclDoubleAllGatherCustom(
    const void *sendBuf0,
    void *recvBuf0,
    uint64_t sendCount0,
    HcclDataType dataType0,
    const void *sendBuf1,
    void *recvBuf1,
    uint64_t sendCount1,
    HcclDataType dataType1,
    HcclComm comm,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

