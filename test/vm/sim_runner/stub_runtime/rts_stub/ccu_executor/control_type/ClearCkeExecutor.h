/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- clear cke
 * Author: z00445483
 */

#ifndef HCCL_SIM_CLEAR_CKE_EXECUTOR_H
#define HCCL_SIM_CLEAR_CKE_EXECUTOR_H

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

class ClearCkeExecutor : public CcuExecutorBase {
public:
    explicit ClearCkeExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    ClearCkeExecutor() = default;
    ~ClearCkeExecutor() = default;

    void Parser() override;
    void Run() override;
    void Process(CcuResouceManager &ccuResMgr) override;
    std::string Describe() override;

private:
    uint16_t clearType_;
    uint16_t clearCKEId_;
    uint16_t clearMask_;
    uint16_t waitCKEId_;
    uint16_t waitCKEMask_;
};

#endif // HCCL_SIM_CLEAR_CKE_EXECUTOR_H
