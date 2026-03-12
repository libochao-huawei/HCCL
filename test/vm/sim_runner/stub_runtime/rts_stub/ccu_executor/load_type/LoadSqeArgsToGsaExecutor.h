/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load sqe arg to gsa
 * Author: z00445483
 */

#ifndef HCCL_SIM_LOAD_SQE_ARGS_TO_GSA_EXECUTOR_H
#define HCCL_SIM_LOAD_SQE_ARGS_TO_GSA_EXECUTOR_H

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

class LoadSqeArgsToGsaExecutor : public CcuExecutorBase {
public:
    explicit LoadSqeArgsToGsaExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    LoadSqeArgsToGsaExecutor() = default;
    ~LoadSqeArgsToGsaExecutor() = default;

    void Parser() override;
    void Run() override;
    std::string Describe() override;

private:
    uint16_t gsaId_{SimCcuV1::CCU_RESOURCE_GSA_MAX};
    uint16_t sqeArgId_{Hccl::CCU_SQE_ARGS_LEN};
};

#endif // HCCL_SIM_LOAD_SQE_ARGS_TO_GSA_EXECUTOR_H
