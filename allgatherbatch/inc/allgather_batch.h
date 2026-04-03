#ifndef HCCL_ALLGATHERBATCH_API_H
#define HCCL_ALLGATHERBATCH_API_H

#include <cstdint>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// 对外暴露的 item 描述符。
// sendCount 的单位是 dataType 对应的元素个数，而不是字节数。
typedef struct HcclAllGatherItemDef {
    void *sendBuf;
    void *recvBuf;
    uint64_t sendCount;
    HcclDataType dataType;
} HcclAllGatherItem;

// 把多个 AllGather 负载合并成一次自定义算子调用。
// 当前实现主要面向 A3 和同一 superpod 内的多 server 场景。
HcclResult HcclAllGatherBatch(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
