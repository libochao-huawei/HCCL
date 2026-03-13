/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- set cke
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "SetCkeExecutor.h"
#include "CcuExecutorManager.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册SetCkeExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::SETCKE_CODE, SetCkeExecutor)

void SetCkeExecutor::Parser() {
    clearType_     = instr_.v1.setCKE.clearType;
    setCKEId_      = instr_.v1.setCKE.setCKEId;
    setCKEMask_    = instr_.v1.setCKE.setCKEMask;
    waitCKEId_     = instr_.v1.setCKE.waitCKEId;
    waitCKEMask_   = instr_.v1.setCKE.waitCKEMask;
}

void SetCkeExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SetCkeExecutor::Run() {
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SetCKE");
}

std::string SetCkeExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Set CKE[%u:%04x], clearType[%u]",
        waitCKEId_,
        waitCKEMask_,
        setCKEId_,
        setCKEMask_,
        clearType_);
}