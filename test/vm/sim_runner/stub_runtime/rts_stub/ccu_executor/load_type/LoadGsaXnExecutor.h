/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load gsa to xn
 * Author: z00445483
 */

#ifndef HCCL_SIM_LOAD_GSAXN_EXECUTOR_H
#define HCCL_SIM_LOAD_GSAXN_EXECUTOR_H

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

class LoadGsaXnExecutor : public CcuExecutorBase {
public:
    explicit LoadGsaXnExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    LoadGsaXnExecutor() = default;
    ~LoadGsaXnExecutor() = default;

    void Parser() override;
    void Run() override;
    std::string Describe() override;

private:
    uint16_t gsAdId_{0};
    uint16_t gsAmId_{0};
    uint16_t xnId_{0};
};

#endif // HCCL_SIM_LOAD_GSAXN_EXECUTOR_H