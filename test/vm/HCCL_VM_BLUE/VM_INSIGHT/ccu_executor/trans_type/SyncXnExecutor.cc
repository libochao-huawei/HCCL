/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- sync xn
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "SyncXnExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册SyncXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::SYNCXN_CODE, SyncXnExecutor)

void SyncXnExecutor::Parser() {
    rmtXnId_       = instr_.v1.syncXn.rmtXnId;
    locXnId_       = instr_.v1.syncXn.locXnId;
    channelId_     = instr_.v1.syncXn.channelId;
    setRmtCKEId_   = instr_.v1.syncXn.setRmtCKEId;
    setRmtCKEMask_ = instr_.v1.syncXn.setRmtCKEMask;
    clearType_     = instr_.v1.syncXn.clearType;
    setCKEId_      = instr_.v1.syncXn.setCKEId;
    setCKEMask_    = instr_.v1.syncXn.setCKEMask;
    waitCKEId_     = instr_.v1.syncXn.waitCKEId;
    waitCKEMask_   = instr_.v1.syncXn.waitCKEMask;
}

void SyncXnExecutor::Process(CcuResouceManager &ccuResMgr) {
    // 根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(dieId_, channelId_);
    // 将本端的xn内容写到目的端的xn中
    auto locXnValue = ccuResMgr.GetXnValue(rankId_, dieId_, locXnId_);
    ccuResMgr.UpdateXnValue(rmtCcu.first, rmtCcu.second, rmtXnId_, locXnValue);

    // 设置目的端的cke
    SetRmtCKESignal(ccuResMgr, rmtCcu.first, rmtCcu.second, setRmtCKEId_, setRmtCKEMask_);

    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SyncXnExecutor::Run() {
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SyncXn");
}

std::string SyncXnExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Sync locXnId[%u] To rmtXnId[%u] Use "
                              "Channel[%u], Set rmtCKE[%u:%04x], Set "
                              "CKE[%u:%04x], clearType[%u]",
        waitCKEId_,
        waitCKEMask_,
        locXnId_,
        rmtXnId_,
        channelId_,
        setRmtCKEId_,
        setRmtCKEMask_,
        setCKEId_,
        setCKEMask_,
        clearType_);
}