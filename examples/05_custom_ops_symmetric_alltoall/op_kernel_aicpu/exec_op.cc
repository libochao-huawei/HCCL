/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exec_op.h"

namespace ops_hccl_symmetric_alltoall {

static inline uint32_t GetDataTypeSize(HcclDataType dataType)
{
    return SIZE_TABLE[static_cast<uint32_t>(dataType)];
}

HcclResult ExecOp(OpParam &param, AlgResourceCtx* resCtx)
{
    uint32_t rank = param.rank;
    uint32_t rankSize = param.rankSize;
    uint64_t count = param.count;
    HcclDataType dataType = param.dataType;

    uint32_t elemSize = GetDataTypeSize(dataType);
    uint64_t chunkSize = count / rankSize;
    uint64_t dataSize = chunkSize * elemSize;

    HCCL_DEBUG("[ExecOp] rank=%u, rankSize=%u, count=%llu, chunkSize=%llu, dataSize=%llu",
                rank, rankSize, count, chunkSize, dataSize);

    for (uint32_t peerRank = 0; peerRank < rankSize; peerRank++) {
        if (peerRank == rank) {
            continue;
        }

        void* peerRecvBuf = nullptr;
        size_t peerOffset = param.recvOffset + peerRank * dataSize;
        int32_t ret = HcommSymWinGetPeerPointer(param.recvWin, peerOffset, peerRank, &peerRecvBuf);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[ExecOp] Get peer recv buffer failed, peerRank=%u, ret=%d", peerRank, ret);
            return static_cast<HcclResult>(ret);
        }

        void* localData = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(param.sendBuf) + peerRank * dataSize);

        ret = HcommWriteOnThread(resCtx->threadHandle, resCtx->channelHandle, peerRecvBuf, localData, dataSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[ExecOp] Write to peer failed, peerRank=%u, ret=%d", peerRank, ret);
            return static_cast<HcclResult>(ret);
        }

        HCCL_DEBUG("[ExecOp] Sent %llu bytes to rank %u, peerRecvBuf=%p", dataSize, peerRank, peerRecvBuf);
    }

    return HCCL_SUCCESS;
}
}
