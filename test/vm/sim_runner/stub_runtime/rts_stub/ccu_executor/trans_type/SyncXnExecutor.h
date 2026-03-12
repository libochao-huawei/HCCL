/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- sync xn
 * Author: z00445483
 */

#ifndef HCCL_SIM_SYNC_XN_EXECUTOR_H
#define HCCL_SIM_SYNC_XN_EXECUTOR_H

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

class SyncXnExecutor : public CcuExecutorBase {
public:
    explicit SyncXnExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    SyncXnExecutor() = default;
    ~SyncXnExecutor() = default;

    void Parser() override;
    void Run() override;
    void Process(CcuResouceManager &ccuResMgr) override;
    std::string Describe() override;

private:
    uint16_t rmtXnId_;
    uint16_t locXnId_;
    uint16_t channelId_;
    uint16_t setRmtCKEId_;
    uint16_t setRmtCKEMask_;
    uint16_t clearType_;
    uint16_t setCKEId_;
    uint16_t setCKEMask_;
    uint16_t waitCKEId_;
    uint16_t waitCKEMask_;
};

#endif // HCCL_SIM_SYNC_XN_EXECUTOR_H
