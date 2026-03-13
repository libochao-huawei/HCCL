/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- jmp
 * Author: z00445483
 */

#ifndef HCCL_SIM_CCU_SIMULATOR_BASE_H
#define HCCL_SIM_CCU_SIMULATOR_BASE_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "rts_stub.h"
#include "ccuMicrocodeV1.h"
#include "ccu_task_param.h"

enum CcuExecState {EXEC_NORMAL_INSTR, EXEC_LOOPGROUP_INSTR, EXEC_LOOP_INSTR, EXEC_JUMP_INSTR, EXEC_SUCCESS, EXEC_FAIL};
enum ReduceAddDataType {ADD_FP32, ADD_FP16, ADD_BF16, ADD_HIF8, ADD_FP8_E4M3, ADD_FP8_E5M2, ADD_INT8, ADD_UINT8, ADD_INT16, ADD_INT32, ADD_RESERVED};
enum ReduceMaxMinDataType {
    MAX_MIN_FP32,
    MAX_MIN_FP16,
    MAX_MIN_BF16,
    MAX_MIN_RESERVED1,
    MAX_MIN_RESERVED2,
    MAX_MIN_RESERVED3,
    MAX_MIN_INT8,
    MAX_MIN_UINT8,
    MAX_MIN_INT16,
    MAX_MIN_INT32,
    MAX_MIN_RESERVED4
};

struct LoopStatusInfo {
    uint16_t loopCurInstrId{0};
    uint16_t loopStartInstrId{0};
    uint16_t loopEndInstrId{0};
    uint16_t loopExecCount{0};
    uint16_t curLoopRound{0};   // 用于记录当前loop循环的执行轮次
    uint32_t loopGsaIterStep{0}; // loop迭代时，gsa地址的偏移步长
    uint16_t loopExtendIndex{0}; // loop的扩展index
};

struct LoopGroupInfo {
    uint16_t startLoopId_{0};
    uint16_t loopNum_{0};
    uint16_t loopOffset_{0};
    uint16_t loopExtendNum_{0};
    uint32_t gsaOffset_{0};     // loop指令执行过程中，gsa地址的偏移量
    uint16_t msOffset_{0};      // loop指令执行过程中，ms的偏移量
    uint16_t ckeOffset_{0};     // loop指令执行过程中，cke的偏移量
    LoopStatusInfo loopStatus_; // loop指令执行状态信息
};

struct ExecuteStatusInfo {
    CcuExecState state{CcuExecState::EXEC_NORMAL_INSTR}; // 当前ccu的执行状态
    uint16_t curInstrId;
    uint16_t startInstrId;
    uint16_t endInstrId;
    uint16_t jumpInstrId;
    uint16_t instrType;  // 当前指令的类型(主要用于记录当前执行是否为Loop)
    LoopGroupInfo loopGroupState;
    bool waitCKE{false}; // 是否需要等待CKE
    bool initialized{false}; // 是否已经初始化
};

#endif // HCCL_SIM_CCU_SIMULATOR_BASE_H
