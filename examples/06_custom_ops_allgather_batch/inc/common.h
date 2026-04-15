/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_ALLGATHER_BATCH_COMMON_H
#define OPS_HCCL_CUSTOM_ALLGATHER_BATCH_COMMON_H

#include "all_gather_batch_item.h"
#include "hccl/hccl_res.h"
#include "acl/acl_rt.h"
#include "log.h"

namespace ops_hccl_allgather_batch {

constexpr uint32_t CUSTOM_TIMEOUT = 1800;
constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t MAX_RANK_SIZE = 8;
constexpr uint32_t MAX_ITEM_COUNT = 8;

struct OpParam {
    char tag[TAG_LENGTH] = {};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};

    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t itemCount = 0;
    HcclAllGatherItem items[MAX_ITEM_COUNT] = {};
};

inline uint32_t GetDataTypeSize(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case HCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case HCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case HCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case HCCL_DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case HCCL_DATA_TYPE_UINT16:
            return sizeof(uint16_t);
        case HCCL_DATA_TYPE_UINT32:
            return sizeof(uint32_t);
        case HCCL_DATA_TYPE_UINT64:
            return sizeof(uint64_t);
        case HCCL_DATA_TYPE_FP16:
            return sizeof(uint16_t);
        case HCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case HCCL_DATA_TYPE_FP64:
            return sizeof(double);
        case HCCL_DATA_TYPE_BFP16:
            return sizeof(uint16_t);
        case HCCL_DATA_TYPE_INT128:
            return 16;
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}

inline bool IsSupportedDataType(HcclDataType dataType)
{
    return GetDataTypeSize(dataType) != 0;
}

} // namespace ops_hccl_allgather_batch

#endif
