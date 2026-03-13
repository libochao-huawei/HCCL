#ifndef HCCLV2_CCU_TRANSFORM_TASK_H
#define HCCLV2_CCU_TRANSFORM_TASK_H

#include <map>
#include <set>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <hccl/hccl_types.h>

#include "base.h"
#include "log.h"
#include "checker_data_slice.h"
#include "ccu_microcode.h"
#include "task_ccu.h"
#include "ccu_instr_info.h"
#include "task_stub.h"
#include "ccu_error_handler.h"
#include "alg_adapt_v2_interface.h"

using namespace checker;
namespace Hccl {

HcclResult GenCcuGraph(TaskNode* dummyStart);

HcclResult TransformInstr(const CcuRep::CcuInstr *instr, uint32_t rankId, uint32_t queId, TaskNode* &preNode, bool& isContinue);

HcclResult TransformInstr(const CcuRep::CcuInstrInfo& instrInfo, uint32_t rankId, uint32_t queId, TaskNode* &preNode, bool isFinished);

HcclResult GetHcclDataTypeFromCCUDataType(uint16_t ccuDataType, uint16_t ccuReduceType, DataType& dataType);

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