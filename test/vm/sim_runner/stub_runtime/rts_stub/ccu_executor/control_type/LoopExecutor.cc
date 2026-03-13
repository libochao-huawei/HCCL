/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- loop
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoopExecutor.h"
#include "CcuExecutorManager.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册SyncXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::LOOP_CODE, LoopExecutor)

void LoopExecutor::Parser() {
    startInstrId_ = instr_.v1.loop.startInstrId;
    endInstrId_   = instr_.v1.loop.endInstrId;
    xnId_         = instr_.v1.loop.xnId;
}

void LoopExecutor::Run() {
    uint64_t xnValue = CcuResouceManager::GetInstance().GetXnValue(rankId_, dieId_, xnId_);
    uint16_t loopCnt = xnValue & 0x1FFF; // 0x1FFF: 取低13位
    // [搬运任务]用到的GSA地址的偏移步长
    uint32_t gsaAddrStep = (xnValue >> 13) & 0xFFFFFFFF; // 13: 右移13位 0xFFFFFFFF: 取低32位

    ccuSimulator_->InitLoopInfo(startInstrId_, endInstrId_, loopCnt, gsaAddrStep);
    ccuSimulator_->ExecuteLoop();
}

std::string LoopExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] Loop From startInstrId[%u] to endInstrId[%u] with loopXn[%u]", startInstrId_, endInstrId_, xnId_);
}