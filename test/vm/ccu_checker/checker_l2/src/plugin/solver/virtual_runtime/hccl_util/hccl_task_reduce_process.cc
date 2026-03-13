#include <iostream>
#include "hccl_task_reduce_process.h"

// 本地任务定义
namespace VirtualRunTime {
// 对不同类型的做Sum的Reduce操作
template <typename T>
void MemReduceSumTemplate(void *dst_addr, const void *src_addr, uint64_t count)
{
    T *dst = (T *)(dst_addr);
    T *src = (T *)(src_addr);
    for (auto index = 0; index < count; ++index) {
        dst[index] += src[index];
    }
}

// 对不同类型的做MIN的Reduce操作
template <typename T>
void MemReduceMinTemplate(void *dst_addr, const void *src_addr, uint64_t count)
{
    T *dst = (T *)(dst_addr);
    T *src = (T *)(src_addr);
    for (auto index = 1; index < count; ++index) {
        if (src[index] < dst[index]) {
            dst[index] = src[index];
        }
    }
}

// 对不同类型的做MAX的Reduce操作
template <typename T>
void MemReduceMaxTemplate(void *dst_addr, const void *src_addr, uint64_t count)
{
    T *dst = (T *)(dst_addr);
    T *src = (T *)(src_addr);
    for (auto index = 1; index < count; ++index) {
        if (src[index] > dst[index]) {
            dst[index] = src[index];
        }
    }
}

void MemReduceSum(void* src, void* dst, uint32_t dataCount, HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return MemReduceSumTemplate<int8_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return MemReduceSumTemplate<int16_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return MemReduceSumTemplate<int32_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return MemReduceSumTemplate<int64_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return MemReduceSumTemplate<uint8_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return MemReduceSumTemplate<uint16_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return MemReduceSumTemplate<uint32_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return MemReduceSumTemplate<uint64_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            // return MemReduceSumTemplate<FP16>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return MemReduceSumTemplate<float>(dst, src, dataCount);
    }
}

void MemReduceMin(void* src, void* dst, uint32_t dataCount, HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return MemReduceMinTemplate<int8_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return MemReduceMinTemplate<int16_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return MemReduceMinTemplate<int32_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return MemReduceMinTemplate<int64_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return MemReduceMinTemplate<uint8_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return MemReduceMinTemplate<uint16_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return MemReduceMinTemplate<uint32_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return MemReduceMinTemplate<uint64_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            // return MemReduceMinTemplate<FP16>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return MemReduceMinTemplate<float>(dst, src, dataCount);
    }
}

void MemReduceMax(void* src, void* dst, uint32_t dataCount, HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return MemReduceMaxTemplate<int8_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return MemReduceMaxTemplate<int16_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return MemReduceMaxTemplate<int32_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return MemReduceMaxTemplate<int64_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return MemReduceMaxTemplate<uint8_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return MemReduceMaxTemplate<uint16_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return MemReduceMaxTemplate<uint32_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return MemReduceMaxTemplate<uint64_t>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            // return MemReduceMaxTemplate<FP16>(dst, src, dataCount);
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return MemReduceMaxTemplate<float>(dst, src, dataCount);
    }
}

}