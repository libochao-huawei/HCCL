/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- set cke
 * Author: z00445483
 */

#ifndef HCCL_SIM_SET_CKE_EXECUTOR_H
#define HCCL_SIM_SET_CKE_EXECUTOR_H

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

class SetCkeExecutor : public CcuExecutorBase {
public:
    explicit SetCkeExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    SetCkeExecutor() = default;
    ~SetCkeExecutor() = default;

    void Parser() override;
    void Run() override;
    void Process(CcuResouceManager &ccuResMgr) override;
    std::string Describe() override;

private:
    uint16_t clearType_;
    uint16_t setCKEId_;
    uint16_t setCKEMask_;
    uint16_t waitCKEId_;
    uint16_t waitCKEMask_;
};

#endif // HCCL_SIM_SET_CKE_EXECUTOR_H
