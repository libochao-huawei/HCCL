#ifndef HCCL_CUSTOM_ALLGATHER_BATCH_CCU_SIGNATURE_TYPES_H
#define HCCL_CUSTOM_ALLGATHER_BATCH_CCU_SIGNATURE_TYPES_H

#include "hccl_types.h"
#include "hccl.h"

namespace ops_hccl {

enum class OpExecuteConfig {
    DEFAULT = 0,
    HOSTCPU_TS = 1,
    AICPU_TS = 2,
    AIV = 3,
    AIV_ONLY = 4,
    CCU_MS = 5,
    CCU_SCHED = 6,
    AICPU = 7,
    HOSTCPU = 8,
    CCU_FAIL
};

struct OpParam {
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    uint32_t root = 0;
    ::CommEngine engine = ::CommEngine::COMM_ENGINE_RESERVED;
    union {
        struct {
            uint64_t count;
            HcclDataType dataType;
            HcclDataType outputType;
            uint64_t strideCount;
        } DataDes = {0, HCCL_DATA_TYPE_RESERVED, HCCL_DATA_TYPE_RESERVED, 0};
        struct {
            void *counts;
            void *displs;
            HcclDataType dataType;
        } vDataDes;
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    OpExecuteConfig opExecuteConfig = OpExecuteConfig::DEFAULT;
};

constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t), sizeof(int16_t), sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t),
    sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1
};

} // namespace ops_hccl

#endif
