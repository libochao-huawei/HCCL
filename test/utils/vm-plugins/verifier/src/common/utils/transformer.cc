/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transformer.h"

std::map<CheckerOpType, HcclCMDType> g_CheckerOpType2HcclCMDType = {
    {CheckerOpType::SCATTER, HcclCMDType::HCCL_CMD_SCATTER}
};

std::map<CheckerReduceOp, HcclReduceOp> g_CheckerReduceOp2HcclReduceOp = {
    {CheckerReduceOp::REDUCE_SUM, HcclReduceOp::HCCL_REDUCE_SUM},
    {CheckerReduceOp::REDUCE_PROD, HcclReduceOp::HCCL_REDUCE_PROD},
    {CheckerReduceOp::REDUCE_MAX, HcclReduceOp::HCCL_REDUCE_MAX},
    {CheckerReduceOp::REDUCE_MIN, HcclReduceOp::HCCL_REDUCE_MIN},
    {CheckerReduceOp::REDUCE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED}
};

std::map<CheckerDataType, HcclDataType> g_CheckerDataType2HcclDataType = {
    {CheckerDataType::DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
    {CheckerDataType::DATA_TYPE_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
    {CheckerDataType::DATA_TYPE_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
    {CheckerDataType::DATA_TYPE_FP16, HcclDataType::HCCL_DATA_TYPE_FP16},
    {CheckerDataType::DATA_TYPE_FP32, HcclDataType::HCCL_DATA_TYPE_FP32},
    {CheckerDataType::DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_INT64},
    {CheckerDataType::DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_UINT64},
    {CheckerDataType::DATA_TYPE_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {CheckerDataType::DATA_TYPE_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {CheckerDataType::DATA_TYPE_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {CheckerDataType::DATA_TYPE_FP64, HcclDataType::HCCL_DATA_TYPE_FP64},
    {CheckerDataType::DATA_TYPE_BFP16, HcclDataType::HCCL_DATA_TYPE_BFP16},
    {CheckerDataType::DATA_TYPE_INT128, HcclDataType::HCCL_DATA_TYPE_INT128},
    {CheckerDataType::DATA_TYPE_HIF8, HcclDataType::HCCL_DATA_TYPE_HIF8},
    {CheckerDataType::DATA_TYPE_FP8E4M3, HcclDataType::HCCL_DATA_TYPE_FP8E4M3},
    {CheckerDataType::DATA_TYPE_FP8E5M2, HcclDataType::HCCL_DATA_TYPE_FP8E5M2},
    {CheckerDataType::DATA_TYPE_RESERVED, HcclDataType::HCCL_DATA_TYPE_RESERVED}
};

std::map<CheckerDataType, uint32_t> g_CheckerDataTypeSize = {
    {CheckerDataType::DATA_TYPE_INT8, 1},
    {CheckerDataType::DATA_TYPE_INT16, 2},
    {CheckerDataType::DATA_TYPE_INT32, 4},
    {CheckerDataType::DATA_TYPE_FP16, 2},
    {CheckerDataType::DATA_TYPE_FP32, 4},
    {CheckerDataType::DATA_TYPE_INT64, 8},
    {CheckerDataType::DATA_TYPE_UINT64, 8},
    {CheckerDataType::DATA_TYPE_UINT8, 1},
    {CheckerDataType::DATA_TYPE_UINT16, 2},
    {CheckerDataType::DATA_TYPE_UINT32, 4},
    {CheckerDataType::DATA_TYPE_FP64, 8},
    {CheckerDataType::DATA_TYPE_BFP16, 2},
    {CheckerDataType::DATA_TYPE_INT128, 16},
    {CheckerDataType::DATA_TYPE_HIF8, 1},
    {CheckerDataType::DATA_TYPE_FP8E4M3, 1},
    {CheckerDataType::DATA_TYPE_FP8E5M2, 1},
    {CheckerDataType::DATA_TYPE_RESERVED, 0}
};

std::vector<u32> sizeTable = {sizeof(s8), sizeof(s16), sizeof(s32), 2, sizeof(float), sizeof(s64), sizeof(u64),
    sizeof(u8), sizeof(u16), sizeof(u32), 8, 2, 16, 2, 1, 1, 1, 1};

std::map<HcclReduceOp, CheckerReduceOp> g_HcclReduceOp2CheckerReduceOp = {
    {HcclReduceOp::HCCL_REDUCE_SUM, CheckerReduceOp::REDUCE_SUM},
    {HcclReduceOp::HCCL_REDUCE_PROD, CheckerReduceOp::REDUCE_PROD},
    {HcclReduceOp::HCCL_REDUCE_MAX, CheckerReduceOp::REDUCE_MAX},
    {HcclReduceOp::HCCL_REDUCE_MIN, CheckerReduceOp::REDUCE_MIN},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, CheckerReduceOp::REDUCE_RESERVED}
};

std::map<HcclDataType, CheckerDataType> g_HcclDataType2CheckerDataType = {
    {HcclDataType::HCCL_DATA_TYPE_INT8, CheckerDataType::DATA_TYPE_INT8},
    {HcclDataType::HCCL_DATA_TYPE_INT16, CheckerDataType::DATA_TYPE_INT16},
    {HcclDataType::HCCL_DATA_TYPE_INT32, CheckerDataType::DATA_TYPE_INT32},
    {HcclDataType::HCCL_DATA_TYPE_FP16, CheckerDataType::DATA_TYPE_FP16},
    {HcclDataType::HCCL_DATA_TYPE_FP32, CheckerDataType::DATA_TYPE_FP32},
    {HcclDataType::HCCL_DATA_TYPE_INT64, CheckerDataType::DATA_TYPE_INT64},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, CheckerDataType::DATA_TYPE_UINT64},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, CheckerDataType::DATA_TYPE_UINT8},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, CheckerDataType::DATA_TYPE_UINT16},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, CheckerDataType::DATA_TYPE_UINT32},
    {HcclDataType::HCCL_DATA_TYPE_FP64, CheckerDataType::DATA_TYPE_FP64},
    {HcclDataType::HCCL_DATA_TYPE_BFP16, CheckerDataType::DATA_TYPE_BFP16},
    {HcclDataType::HCCL_DATA_TYPE_INT128, CheckerDataType::DATA_TYPE_INT128},
    {HcclDataType::HCCL_DATA_TYPE_HIF8, CheckerDataType::DATA_TYPE_HIF8},
    {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, CheckerDataType::DATA_TYPE_FP8E4M3},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, CheckerDataType::DATA_TYPE_FP8E5M2},
    {HcclDataType::HCCL_DATA_TYPE_RESERVED, CheckerDataType::DATA_TYPE_RESERVED}
};