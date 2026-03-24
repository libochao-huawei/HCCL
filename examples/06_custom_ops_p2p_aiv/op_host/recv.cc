/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_custom_p2p_aiv.h"

#include "common.h"
#include "resource.h"

using namespace ops_hccl_p2p_aiv;

extern "C" HcclResult HcclRecvCustomAiv(
    void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    uint64_t typeBytes = 0;
    CHK_RET(GetDataTypeBytes(dataType, &typeBytes));
    CHK_PRT_RET(typeBytes != 0 && count > (UINT64_MAX / typeBytes),
        HCCL_ERROR("count overflow, count=%llu typeBytes=%llu",
            static_cast<unsigned long long>(count), static_cast<unsigned long long>(typeBytes)), HCCL_E_PARA);
    const uint64_t lenBytes = count * typeBytes;

    P2pAivResource resource;
    CHK_RET(PrepareP2pAivResource(comm, srcRank, "hccl_recv_custom_aiv", stream, &resource));
    return ExecuteRecv(resource, recvBuf, lenBytes, dataType, stream);
}
