/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_SYMMETRIC_ALLTOALL_H_
#define HCCL_CUSTOM_SYMMETRIC_ALLTOALL_H_

#include <hccl/hccl_types.h>
#include <acl/acl.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Symmetric AlltoAll operator.
 *
 * This operator performs an all-to-all communication where each rank sends
 * data to all other ranks and receives data from all other ranks. It uses
 * symmetric memory (registered via HcclCommSymWinRegister) to achieve zero-copy.
 *
 * @param sendBuf A pointer to the send buffer (must be registered symmetric memory).
 * @param recvBuf A pointer to the receive buffer (must be registered symmetric memory).
 * @param count The total number of elements to send to each rank (count/rankSize per peer).
 * @param dataType The data type of the elements.
 * @param comm The HCCL communicator.
 * @param stream The ACL stream.
 * @return HcclResult
 */
extern HcclResult HcclAllToAllCustom(
    void* sendBuf,
    uint64_t count,
    HcclDataType dataType,
    void* recvBuf,
    HcclComm comm,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // HCCL_CUSTOM_SYMMETRIC_ALLTOALL_H_
