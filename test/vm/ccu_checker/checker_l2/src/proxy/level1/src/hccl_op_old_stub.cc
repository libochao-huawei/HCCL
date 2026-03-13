/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
* Inner后缀的接口为算子旧流程封装, 当前阶段打桩都不实现
*/

HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support\n", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAllReduceInner(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclBroadcastInner(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclReduceScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclReduceScatterVInner(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf,
    uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAllGatherInner(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
    aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAllGatherVInner(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
    const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclSendInner(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm,
    aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclRecvInner(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm,
    aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAlltoAllVCInner(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAlltoAllVInner(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, HcclComm comm,
    aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAlltoAllInner(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclReduceInner(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclBatchSendRecvInner(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    printf("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

#ifdef __cplusplus
}
#endif  // __cplusplus