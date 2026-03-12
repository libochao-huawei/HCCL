/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor base header file
 * Author: z00445483
 */

#ifndef HCCL_SIM_CCU_EXECUTOR_BASE_H
#define HCCL_SIM_CCU_EXECUTOR_BASE_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "rts_stub.h"
#include "FakeStreamMgr.h"
#include "ccu_microcode.h"
#include "ccuMicrocodeV1.h"
#include "CcuResourceManager.h"

class CcuExecutorBase {
public:
    CcuExecutorBase(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : streamId_(streamId), rankId_(rankId), dieId_(dieId), instr_(instr), ccuSimulator_(ccuSimulator)
    {}
    virtual ~CcuExecutorBase() = default;

    virtual void Parser() = 0;
    virtual void Run() = 0;
    virtual std::string Describe() = 0;
    virtual void Process(CcuResouceManager &ccuResMgr) { return; }

    std::string ParseMSList();
    void SetCkeSignal(CcuResouceManager &ccuResMgr, uint16_t setCKEId, uint16_t setCKEMask);
    void SetRmtCKESignal(CcuResouceManager &ccuResMgr, int rmtRank, int rmtDie, uint16_t setRmtCKEId, uint16_t setRmtCKEMask);
    void ClearCkeSignal(CcuResouceManager &ccuResMgr, uint16_t clearCKEId, uint16_t clearMask);
    void WaitCkeProcess(uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, const std::string &instrName);

public:
    int rankId_{0};
    int dieId_{0};
    int streamId_{0};
    Hccl::CcuRep::CcuInstr instr_;  // 指令信息
    CcuSimulator *ccuSimulator_{nullptr}; // ccu指令模拟器
};

#endif // HCCL_SIM_CCU_EXECUTOR_BASE_H
