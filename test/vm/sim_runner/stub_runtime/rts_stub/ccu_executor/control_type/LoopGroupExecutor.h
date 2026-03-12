/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- loop group
 * Author: z00445483
 */

#ifndef HCCL_SIM_LOOP_GROUP_EXECUTOR_H
#define HCCL_SIM_LOOP_GROUP_EXECUTOR_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "rts_stub.h"
#include "FakeStreamMgr.h"
#include "ccuMicrocodeV1.h"
#include "ccu_task_param.h"
#include "CcuExecutorBase.h"
#include "CcuResourceManager.h"

class LoopGroupExecutor : public CcuExecutorBase {
public:
    explicit LoopGroupExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    LoopGroupExecutor() = default;
    ~LoopGroupExecutor() = default;

    void Parser() override;
    void Run() override;
    std::string Describe() override;

private:
    // bool ExecuteInstr(CcuResouceManager &ccuResMgr, int devId, uint16_t curInstrId);

private:
    uint16_t startLoopInstrId_;
    uint16_t xnId_;
    uint16_t xmId_;
    uint16_t highPerfModeEn_;
};

#endif // HCCL_SIM_LOOP_GROUP_EXECUTOR_H
