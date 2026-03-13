/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- reduce max
 * Author: z00445483
 */

#ifndef HCCL_SIM_REDUCE_MAX_EXECUTOR_H
#define HCCL_SIM_REDUCE_MAX_EXECUTOR_H

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

class ReduceMaxExecutor : public CcuExecutorBase {
public:
    explicit ReduceMaxExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {
        (void)memset_s(msId_, sizeof(uint16_t) * Hccl::CcuRep::CCU_REDUCE_MAX_MS, 0, sizeof(uint16_t) * Hccl::CcuRep::CCU_REDUCE_MAX_MS);
    }
    ReduceMaxExecutor() = default;
    ~ReduceMaxExecutor() = default;

    void Parser() override;
    void Run() override;
    void Process(CcuResouceManager &ccuResMgr) override;
    std::string Describe() override;

private:
    uint16_t count_{0};
    uint16_t dataType_{0};
    uint16_t clearType_{0};
    uint16_t setCKEId_{0};
    uint16_t setCKEMask_{0};
    uint16_t waitCKEId_{0};
    uint16_t waitCKEMask_{0};
    uint16_t msId_[Hccl::CcuRep::CCU_REDUCE_MAX_MS];
};

#endif // HCCL_SIM_REDUCE_MAX_EXECUTOR_H