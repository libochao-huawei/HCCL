#ifndef HCCL_CUSTOM_ALLGATHER_2IN2OUT_H
#define HCCL_CUSTOM_ALLGATHER_2IN2OUT_H

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclAllGather2In2OutAicpuCustom(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
