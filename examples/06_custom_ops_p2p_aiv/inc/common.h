/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_AIV_COMMON_H
#define OPS_HCCL_P2P_AIV_COMMON_H

#include <cstdint>

#include "acl/acl_rt.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hcomm_primitives.h"
#include "kernel_types.h"
#include "log.h"

namespace ops_hccl_p2p_aiv {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t COMM_IDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_IDENTIFIER_MAX_LENGTH;
constexpr uint32_t MAX_NET_LAYER = 3;

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct P2pAivResource {
    ThreadHandle threadHandle = 0;
    ChannelHandle channelHandle = 0;
    CommBuffer localBuffer;
    CommBuffer remoteBuffer;
    void *syncCtx = nullptr;
    uint64_t syncCtxSize = kP2pAivSyncCtxSize;
    void *remoteSyncCtx = nullptr;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t peerRank = 0;
    char commName[COMM_IDENTIFIER_MAX_LENGTH] = {0};
    char tag[TAG_LENGTH] = {0};
};

inline HcclResult GetDataTypeBytes(HcclDataType dataType, uint64_t *typeBytes)
{
    CHK_PTR_NULL(typeBytes);
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
            *typeBytes = sizeof(uint8_t);
            break;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            *typeBytes = sizeof(uint16_t);
            break;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            *typeBytes = sizeof(uint32_t);
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            *typeBytes = sizeof(uint64_t);
            break;
        case HCCL_DATA_TYPE_INT128:
            *typeBytes = 16;
            break;
        default:
            HCCL_ERROR("unsupported dataType=%d", dataType);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_p2p_aiv

#endif // OPS_HCCL_P2P_AIV_COMMON_H
