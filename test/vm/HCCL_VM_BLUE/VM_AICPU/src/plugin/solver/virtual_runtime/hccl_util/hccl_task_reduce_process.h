#ifndef HCCL_TASK_REDUCE_PROCESS_H
#define HCCL_TASK_REDUCE_PROCESS_H

#include <thread>
#include <vector>
#include "hccl_common_defs.h"

namespace VirtualRunTime {

    typedef enum {  // todo 临时规避，最好引用HCCL业务代码的头文件
        HCCL_DATA_TYPE_INT8 = 0,    /**< int8 */
        HCCL_DATA_TYPE_INT16 = 1,   /**< int16 */
        HCCL_DATA_TYPE_INT32 = 2,   /**< int32 */
        HCCL_DATA_TYPE_FP16 = 3,    /**< fp16 */
        HCCL_DATA_TYPE_FP32 = 4,    /**< fp32 */
        HCCL_DATA_TYPE_INT64 = 5,    /**< int64 */
        HCCL_DATA_TYPE_UINT64 = 6,    /**< uint64 */
        HCCL_DATA_TYPE_UINT8 = 7,    /**< uint8 */
        HCCL_DATA_TYPE_UINT16 = 8,   /**< uint16 */
        HCCL_DATA_TYPE_UINT32 = 9,   /**< uint32 */
        HCCL_DATA_TYPE_FP64 = 10,    /**< fp64 */
        HCCL_DATA_TYPE_BFP16 = 11,    /**< bfp16 */
        HCCL_DATA_TYPE_INT128 = 12,   /**< int128 */
    #ifndef OPEN_BUILD_PROJECT
        HCCL_DATA_TYPE_HIF8 = 14,     /**< hif8 */
        HCCL_DATA_TYPE_FP8E4M3 = 15,  /**< fp8e4m3 */
        HCCL_DATA_TYPE_FP8E5M2 = 16,  /**< fp8e5m2 */
        HCCL_DATA_TYPE_FP8E8M0 = 17,  /**< fp8e8m0 */
    #endif
        HCCL_DATA_TYPE_RESERVED = 255 /**< reserved */
    } HcclDataType;

    void MemReduceSum(void* src, void* dst, uint32_t dataCount, HcclDataType dataType);
    void MemReduceMin(void* src, void* dst, uint32_t dataCount, HcclDataType dataType);
    void MemReduceMax(void* src, void* dst, uint32_t dataCount, HcclDataType dataType);
}

#endif