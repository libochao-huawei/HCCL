/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: data type header file
 * Author: zhangqingwei
 * Create: 2024-02-04
 */

#ifndef HCCLV2_DATA_TYPE_H
#define HCCLV2_DATA_TYPE_H

#include <map>
#include "enum_factory.h"
#include <hccl_types.h>


namespace HcclSim {

MAKE_ENUM(DataType, INT8, INT16, INT32, FP16, FP32, INT64, UINT64, UINT8, UINT16, UINT32, 
            FP64, BFP16, INT128, BF16_SAT, HIF8, FP8E4M3, FP8E5M2, FP8E8M0)

const std::map<DataType, u32> DATA_TYPE_SIZE_MAP = {
    {DataType::INT8, sizeof(s8)},
    {DataType::INT16, sizeof(s16)},
    {DataType::INT32, sizeof(s32)},
    {DataType::FP16, 2},
    {DataType::FP32, sizeof(float)},
    {DataType::INT64, sizeof(s64)},
    {DataType::UINT64, sizeof(u64)},
    {DataType::UINT8, sizeof(u8)},
    {DataType::UINT16, sizeof(u16)},
    {DataType::UINT32, sizeof(u32)},
    {DataType::FP64, 8},
    {DataType::BFP16, 2},
    {DataType::INT128, 16},
    {DataType::BF16_SAT, 2},
    {DataType::HIF8, 1},
    {DataType::FP8E4M3, 1},
    {DataType::FP8E5M2, 1},
    {DataType::FP8E8M0, 1}
};

const std::map<DataType, HcclDataType> HCCL_DATA_TYPE_MAP = {
    {DataType::INT8, HCCL_DATA_TYPE_INT8},
    {DataType::INT16, HCCL_DATA_TYPE_INT16},
    {DataType::INT32, HCCL_DATA_TYPE_INT32},
    {DataType::FP16, HCCL_DATA_TYPE_FP16},
    {DataType::FP32, HCCL_DATA_TYPE_FP32},
    {DataType::INT64, HCCL_DATA_TYPE_INT64},
    {DataType::UINT64, HCCL_DATA_TYPE_UINT64},
    {DataType::UINT8, HCCL_DATA_TYPE_UINT8},
    {DataType::UINT16, HCCL_DATA_TYPE_UINT16},
    {DataType::UINT32, HCCL_DATA_TYPE_UINT32},
    {DataType::FP64, HCCL_DATA_TYPE_FP64},
    {DataType::BFP16, HCCL_DATA_TYPE_BFP16},
    {DataType::INT128, HCCL_DATA_TYPE_INT128},
    {DataType::HIF8, HCCL_DATA_TYPE_HIF8},
    {DataType::FP8E4M3, HCCL_DATA_TYPE_FP8E4M3},
    {DataType::FP8E5M2, HCCL_DATA_TYPE_FP8E5M2},
    {DataType::FP8E8M0, HCCL_DATA_TYPE_FP8E8M0},
};

const std::map<HcclDataType, DataType> DATA_TYPE_MAP = {
    {HCCL_DATA_TYPE_INT8, DataType::INT8},
    {HCCL_DATA_TYPE_INT16, DataType::INT16},
    {HCCL_DATA_TYPE_INT32, DataType::INT32},
    {HCCL_DATA_TYPE_FP16, DataType::FP16},
    {HCCL_DATA_TYPE_FP32, DataType::FP32},
    {HCCL_DATA_TYPE_INT64, DataType::INT64},
    {HCCL_DATA_TYPE_UINT64, DataType::UINT64},
    {HCCL_DATA_TYPE_UINT8, DataType::UINT8},
    {HCCL_DATA_TYPE_UINT16, DataType::UINT16},
    {HCCL_DATA_TYPE_UINT32, DataType::UINT32},
    {HCCL_DATA_TYPE_FP64, DataType::FP64},
    {HCCL_DATA_TYPE_BFP16, DataType::BFP16},
    {HCCL_DATA_TYPE_INT128, DataType::INT128},
    {HCCL_DATA_TYPE_HIF8, DataType::HIF8},
    {HCCL_DATA_TYPE_FP8E4M3, DataType::FP8E4M3},
    {HCCL_DATA_TYPE_FP8E5M2, DataType::FP8E5M2},
    {HCCL_DATA_TYPE_FP8E8M0, DataType::FP8E8M0},
};

const std::map<uint32_t, std::string> DATA_TYPE_TO_STRING_MAP
    = {{0, "INT8"}, {1, "INT16"}, {2, "INT32"}, {3, "FP16"}, {4, "FP32"}, {5, "INT64"}};

const std::map<uint32_t, std::string> OP_TYPE_TO_STRING_MAP = {{0, "sum"}, {1, "mul"}, {2, "max"}, {3, "PROD"}};

} // namespace Hccl

#endif // HCCLV2_DATA_TYPE_H