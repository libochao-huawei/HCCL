#ifndef HCCLV2_CCU_TRANSFORM_TASK_H
#define HCCLV2_CCU_TRANSFORM_TASK_H

#include <map>
#include <set>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <hccl_types.h>

#include "base.h"
#include "log.h"
#include "data_slice.h"
#include "ccu_microcode.h"
#include "task_ccu.h"
#include "ccu_instr_info.h"
#include "sim_task.h"
#include "task_def.h"
// #include "ccu_error_handler.h"
// #include "alg_adapt_v2_interface.h"
#include "data_type.h"

using namespace hcomm;
namespace HcclSim {

HcclResult GenCcuGraph(TaskNode* dummyStart);

HcclResult TransformInstr(const CcuRep::CcuInstr *instr, uint32_t rankId, uint32_t queId, TaskNode* &preNode, bool& isContinue);

HcclResult TransformInstr(const CcuRep::CcuInstrInfo& instrInfo, uint32_t rankId, uint32_t queId, TaskNode* &preNode, bool isFinished);

HcclResult GetHcclDataTypeFromCCUDataType(uint16_t ccuDataType, uint16_t ccuReduceType, DataType& dataType);

union LoopXm {
    uint64_t value;
    struct {
        uint64_t loopCnt : 13;
        uint64_t gsaStride : 32;
        uint64_t loopCtxId : 8;
        uint64_t reserved : 11;
    };
};

union LoopGroupXn {
    uint64_t value;
    struct {
        uint64_t reservedLow : 41;
        uint64_t loopInsCnt : 7;
        uint64_t expandOffset : 7;
        uint64_t expandCnt : 7;
        uint64_t reservedHigh : 2;
    };
};

union LoopGroupXm {
    uint64_t value;
    struct {
        uint64_t ckOffset : 10;
        uint64_t msOffset : 11;
        uint64_t gsaOffset : 32;
        uint64_t reserved : 11;
    };
};

struct ErrorInfoBase {
    int32_t  deviceId;
    uint8_t  dieId;
    uint8_t  missionId;
    uint16_t currentInsId;
    uint16_t status;
};

struct LoopGroupParam {
    std::vector<LoopXm> loopXms;
    LoopGroupXn loopGroupXn;
    LoopGroupXm loopGroupXm;
    u32 curLoopIdx = 0;                   // 表示当前处理第几个loop
    u32 curExpandCnt = 0;                 // 该loop被第几次展开
    u32 curLoopCnt = 0;                   // 表示当前Loop第几次循环
};

}

#endif