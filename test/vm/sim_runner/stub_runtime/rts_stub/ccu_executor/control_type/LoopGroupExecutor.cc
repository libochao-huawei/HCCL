/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- loop group
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoopGroupExecutor.h"
#include "CcuExecutorManager.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册SyncXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::LOOPGROUP_CODE, LoopGroupExecutor)

void LoopGroupExecutor::Parser() {
    startLoopInstrId_ = instr_.v1.loopGroup.startLoopInstrId;
    xnId_             = instr_.v1.loopGroup.xnId;
    xmId_             = instr_.v1.loopGroup.xmId;
    highPerfModeEn_   = instr_.v1.loopGroup.highPerfModeEn;
}

void LoopGroupExecutor::Run() {
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    uint64_t offsetCfg = ccuResMgr.GetXnValue(rankId_, dieId_, xmId_);
    uint64_t repeatCfg = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
    ccuSimulator_->InitLoopGroupInfo(startLoopInstrId_, offsetCfg, repeatCfg);

    // 执行loop指令
    ccuSimulator_->ExecuteLoopGroup();
    ccuSimulator_->SetExecState(CcuExecState::EXEC_NORMAL_INSTR);
}

std::string LoopGroupExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] LoopGroup From startLoopInstrId[%u] with loopGroupXn[%u], "
                              "offsetXn[%u] and highPerfModeEn[%u]",
        startLoopInstrId_,
        xnId_,
        xmId_,
        highPerfModeEn_);
}