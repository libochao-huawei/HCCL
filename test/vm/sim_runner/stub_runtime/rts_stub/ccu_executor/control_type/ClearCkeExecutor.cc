/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- clear cke
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "ClearCkeExecutor.h"
#include "CcuExecutorManager.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册ClearCkeExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::CTRL_TYPE, SimCcuV1::CLEARCKE_CODE, ClearCkeExecutor)

void ClearCkeExecutor::Parser() {
    clearType_   = instr_.v1.clearCKE.clearType;
    clearCKEId_  = instr_.v1.clearCKE.clearCKEId;
    clearMask_   = instr_.v1.clearCKE.clearMask;
    waitCKEId_   = instr_.v1.clearCKE.waitCKEId;
    waitCKEMask_ = instr_.v1.clearCKE.waitCKEMask;
}

void ClearCkeExecutor::Process(CcuResouceManager &ccuResMgr)
{
    ClearCkeSignal(ccuResMgr, waitCKEId_, waitCKEMask_);
}

void ClearCkeExecutor::Run() {
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "ClearCKE");
}

std::string ClearCkeExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Clear CKE[%u:%04x], clearType[%u]",
        waitCKEId_,
        waitCKEMask_,
        clearCKEId_,
        clearMask_,
        clearType_);
}