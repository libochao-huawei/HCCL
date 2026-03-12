/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- jmp
 * Author: z00445483
 */

#ifndef HCCL_SIM_JUMP_EXECUTOR_H
#define HCCL_SIM_JUMP_EXECUTOR_H

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

class JumpExecutor : public CcuExecutorBase {
public:
    explicit JumpExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    JumpExecutor() = default;
    ~JumpExecutor() = default;

    void Parser() override;
    void Run() override;
    std::string Describe() override;

private:
    uint16_t dstInstrXnId_{0};
    uint16_t conditionXnId_{0};
    uint64_t expectData_{0};
};

#endif // HCCL_SIM_JUMP_EXECUTOR_H
