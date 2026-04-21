#ifndef HCCL_CUSTOM_ALLGATHER_BATCH_CCU_SIGNATURE_TYPES_H
#define HCCL_CUSTOM_ALLGATHER_BATCH_CCU_SIGNATURE_TYPES_H

#include <cstdint>
#include <cstring>

#include "hccl_types.h"
#include "hccl.h"
#include "hccl/hccl_res.h"

namespace ops_hccl {

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t OP_ALG_LENGTH = 128;
constexpr uint32_t ALG_TAG_LENGTH = TAG_LENGTH + OP_ALG_LENGTH;
constexpr uint32_t ALG_MAX_LENGTH = 128;
constexpr uint32_t INVALID_VALUE_RANKID = 0xFFFFFFFF;

enum class OpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

enum class DevType {
    DEV_TYPE_NOSOC = 0,
    DEV_TYPE_910 = 1,
    DEV_TYPE_910B = 2,
    DEV_TYPE_910_93 = 3,
    DEV_TYPE_950 = 4,
    DEV_TYPE_910_95 = 5,
    DEV_TYPE_COUNT
};

enum class AlgTypeLevel0 {
    ALG_LEVEL0_WHOLE_RING = 0,
    ALG_LEVEL0_RESERVED
};

enum class AlgTypeLevel1 {
    ALG_LEVEL1_WHOLE_RING = 0,
    ALG_LEVEL1_RESERVED
};

enum class AlgTypeLevel2 {
    ALG_LEVEL2_WHOLE_RING = 0,
    ALG_LEVEL2_RESERVED
};

struct AlgType {
    AlgTypeLevel0 algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
    AlgTypeLevel1 algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
    AlgTypeLevel2 algoLevel2 = AlgTypeLevel2::ALG_LEVEL2_WHOLE_RING;
};

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
    void *hcclComm = nullptr;
    char tag[TAG_LENGTH] = {};
    char algTag[ALG_TAG_LENGTH] = {};
    char fastLaunchTag[ALG_TAG_LENGTH] = {};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    char commModeTag[TAG_LENGTH] = {};
    aclrtStream stream = nullptr;
    void *inputPtr = nullptr;
    uint64_t inputSize = 0;
    void *outputPtr = nullptr;
    uint64_t outputSize = 0;
    HcclMem hcclBuff{};
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    uint32_t root = INVALID_VALUE_RANKID;
    uint32_t userRank = INVALID_VALUE_RANKID;
    uint32_t sendRecvRemoteRank = INVALID_VALUE_RANKID;
    OpMode opMode = OpMode::OPBASE;
    bool enableDetour = false;
    bool isMc2 = false;
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CommEngine engine = CommEngine::COMM_ENGINE_RESERVED;
    AlgType algType{};
    char algTypeStr[ALG_MAX_LENGTH] = {};
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
        } vDataDes = {nullptr, nullptr, HCCL_DATA_TYPE_RESERVED};
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool isZeroCopy = false;
    char algName[OP_ALG_LENGTH] = {};
    OpExecuteConfig opExecuteConfig = OpExecuteConfig::DEFAULT;
    uint32_t numBlocksLimit = 0;
    bool isAivClearEnable = false;
    uint64_t ctxSize = 0;
    void *resCtx = nullptr;
    ThreadHandle opThread = 0;
    uint32_t aicpuRecordCpuIdx = 0;
    uint32_t dataCount = 0;
    uint64_t varMemSize = 0;
};

constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t), sizeof(int16_t), sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t),
    sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1
};

} // namespace ops_hccl

#endif
