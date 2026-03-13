/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- jmp
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "JumpExecutor.h"
#include "CcuExecutorManager.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册JumpExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::JMP_CODE, JumpExecutor)

void JumpExecutor::Parser()
{
    dstInstrXnId_   = instr_.v1.jmp.dstInstrXnId;
    conditionXnId_  = instr_.v1.jmp.conditionXnId;
    expectData_     = instr_.v1.jmp.expectData;
}

void JumpExecutor::Run()
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    uint64_t instrId = ccuResMgr.GetXnValue(rankId_, dieId_, dstInstrXnId_);
    uint64_t value = ccuResMgr.GetXnValue(rankId_, dieId_, conditionXnId_);

    if (value != expectData_) {
        ccuSimulator_->InitJumpStatus(instrId);
    }
    return;
}

std::string JumpExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] When conditionXn[%u] not equal to expectData[%llu], Jump To InstrIdXn[%u]",
        conditionXnId_, expectData_, dstInstrXnId_);
}