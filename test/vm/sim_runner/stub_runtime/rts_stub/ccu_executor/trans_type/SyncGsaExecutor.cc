/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- sync gsa
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "SyncGsaExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册SyncGsaExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::SYNCGSA_CODE, SyncGsaExecutor)

void SyncGsaExecutor::Parser() {
    rmtGSAId_      = instr_.v1.syncGSA.rmtGSAId;
    locGSAId_      = instr_.v1.syncGSA.locGSAId;
    channelId_     = instr_.v1.syncGSA.channelId;
    setRmtCKEId_   = instr_.v1.syncGSA.setRmtCKEId;
    setRmtCKEMask_ = instr_.v1.syncGSA.setRmtCKEMask;
    clearType_     = instr_.v1.syncGSA.clearType;
    setCKEId_      = instr_.v1.syncGSA.setCKEId;
    setCKEMask_    = instr_.v1.syncGSA.setCKEMask;
    waitCKEId_     = instr_.v1.syncGSA.waitCKEId;
    waitCKEMask_   = instr_.v1.syncGSA.waitCKEMask;
}

void SyncGsaExecutor::Process(CcuResouceManager &ccuResMgr) {
    // 根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(dieId_, channelId_);
    // 获取地址
    auto srcAdrr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    // 将本端的地址写到对端的gsa上
    ccuResMgr.UpdateGsaValue(rmtCcu.first, rmtCcu.second, rmtGSAId_, srcAdrr);

    // 设置目的端的cke
    SetRmtCKESignal(ccuResMgr, rmtCcu.first, rmtCcu.second, setRmtCKEId_, setRmtCKEMask_);

    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SyncGsaExecutor::Run() {
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SyncGsa");
}

std::string SyncGsaExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Sync locGSAId[%u] To rmtGSAId[%u] Use "
                              "Channel[%u], Set rmtCKE[%u:%04x], Set "
                              "CKE[%u:%04x], clearType[%u]",
        waitCKEId_,
        waitCKEMask_,
        locGSAId_,
        rmtGSAId_,
        channelId_,
        setRmtCKEId_,
        setRmtCKEMask_,
        setCKEId_,
        setCKEMask_,
        clearType_);
}