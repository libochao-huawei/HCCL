/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- jmp
 * Author: z00445483
 */

#ifndef HCCL_SIM_CCU_SIMULATOR_H
#define HCCL_SIM_CCU_SIMULATOR_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "rts_stub.h"
#include "ccuMicrocodeV1.h"
#include "ccu_task_param.h"
#include "SimulatorBase.h"

class CcuSimulator {
public:
    explicit CcuSimulator(int rankId, int dieId, uint16_t startInstrId, uint16_t endInstrId, uint16_t instrCnt)
        : rankId_(rankId), dieId_(dieId), curInstrId_(startInstrId), startInstrId_(startInstrId), endInstrId_(endInstrId),
          instrCnt_(instrCnt)
    {}
    CcuSimulator() = default;
    ~CcuSimulator() = default;

    bool Execute();
    bool ExecuteLoop();
    bool ExecuteLoopGroup();
    bool ExecuteInstr(uint16_t curInstrId);
    bool UpdateLoopStatus();

    void SetWaitCKEFlag(bool needCKE);
    void SetExecState(CcuExecState state);

    void Init(uint16_t startInstrId, uint16_t endInstrId, uint16_t instrCnt);
    void InitLoopGroupInfo(const LoopGroupInfo &loopGroupInfo);
    void InitLoopGroupInfo(uint16_t startLoopId, uint64_t offsetCfg, uint64_t repeatCfg);
    void InitLoopInfo(uint16_t startInstrId, uint16_t endInstrId, uint16_t execCount, uint32_t addrBase);
    void InitJumpStatus(uint16_t jumpInstrId);

    uint64_t GetLoopGsaAddrOffset();
    uint16_t GetLoopMsOffset();
    uint16_t GetLoopCKEOffset();

    CcuExecState GetState();

private:
    int rankId_{0};
    int dieId_{0};
    bool finshed_{false};
    bool waitCKE_{false}; // 是否需要等待CKE
    uint16_t curInstrId_;
    uint16_t startInstrId_;
    uint16_t endInstrId_;
    uint16_t instrCnt_;
    uint16_t jumpInstrId_{0};
    uint16_t instrType_{0};  // 当前指令的类型(主要用于记录当前执行是否为Loop)
    bool initialized_{false}; // 是否已经初始化
    CcuExecState state_{CcuExecState::EXEC_NORMAL_INSTR}; // 当前ccu的执行状态

    LoopGroupInfo loopGroupInfo_; // loopGroup指令信息
};

#endif // HCCL_SIM_CCU_SIMULATOR_H
