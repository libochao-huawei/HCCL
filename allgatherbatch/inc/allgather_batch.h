#ifndef HCCL_ALLGATHERBATCH_API_H
#define HCCL_ALLGATHERBATCH_API_H

#include <cstdint>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HcclAllGatherItemDef {
    void *sendBuf;
    void *recvBuf;
    uint64_t sendCount;
    HcclDataType dataType;
} HcclAllGatherItem;

HcclResult HcclAllGatherBatch(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
