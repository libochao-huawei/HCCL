#include <iostream>
#include "dtype_common.h"
#include "hccl_types.h"
#include "hccl_vm_log.h"

namespace sim {

const std::map<HcclDataType, u32> DATA_TYPE_SIZE_MAP = {
    {HcclDataType::HCCL_DATA_TYPE_INT8, sizeof(s8)},
    {HcclDataType::HCCL_DATA_TYPE_INT16, sizeof(s16)},
    {HcclDataType::HCCL_DATA_TYPE_INT32, sizeof(s32)},
    {HcclDataType::HCCL_DATA_TYPE_FP16, 2},
    {HcclDataType::HCCL_DATA_TYPE_FP32, sizeof(float)},
    {HcclDataType::HCCL_DATA_TYPE_INT64, sizeof(s64)},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, sizeof(u64)},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, sizeof(u8)},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, sizeof(u16)},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, sizeof(u32)},
    {HcclDataType::HCCL_DATA_TYPE_FP64, sizeof(double)}, 
    {HcclDataType::HCCL_DATA_TYPE_BFP16, 2},
    {HcclDataType::HCCL_DATA_TYPE_INT128, 16},
    {HcclDataType::HCCL_DATA_TYPE_HIF8, 1},
    {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, 1},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, 1},
    {HcclDataType::HCCL_DATA_TYPE_FP8E8M0, 1}
};

int GetDataTypeSize(HcclDataType dataType, uint32_t &size)
{
    auto iter = DATA_TYPE_SIZE_MAP.find(dataType);
    if (iter == DATA_TYPE_SIZE_MAP.end()) {
        HCCL_VM_ERROR("[GetDataTypeSize] not support data type: {}", static_cast<uint32_t>(dataType));
        return 1;
    }
    size = iter->second;
    return 0;
}

}
