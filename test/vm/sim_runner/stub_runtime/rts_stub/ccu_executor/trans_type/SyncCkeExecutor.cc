/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- sync cke
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "SyncCkeExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册SyncCkeExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::SYNCCKE_CODE, SyncCkeExecutor)

void SyncCkeExecutor::Parser() {
    rmtCKEId_    = instr_.v1.syncCKE.rmtCKEId;
    locCKEId_    = instr_.v1.syncCKE.locCKEId;
    locCKEMask_  = instr_.v1.syncCKE.locCKEMask;
    channelId_   = instr_.v1.syncCKE.channelId;
    clearType_   = instr_.v1.syncCKE.clearType;
    setCKEId_    = instr_.v1.syncCKE.setCKEId;
    setCKEMask_  = instr_.v1.syncCKE.setCKEMask;
    waitCKEId_   = instr_.v1.syncCKE.waitCKEId;
    waitCKEMask_ = instr_.v1.syncCKE.waitCKEMask;
}

void SyncCkeExecutor::Process(CcuResouceManager &ccuResMgr) {
    // 根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(dieId_, channelId_);
    // 将本端的CKE内容写到目的端的CKE中
    auto locCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, locCKEId_);
    uint16_t rmtCke = ccuResMgr.GetCkeValue(rmtCcu.first, rmtCcu.second, rmtCKEId_);
    uint16_t newRmtCke = rmtCke | (locCKE & locCKEMask_);
    ccuResMgr.UpdateCkeValue(rmtCcu.first, rmtCcu.second, rmtCKEId_, newRmtCke);

    HCCL_DEBUG("[SyncCkeExecutor][Process] locCcu[%d:%d], channelId=[%u], rmtDevice[%d:%d], locCKE=[%u:%04x], "
               "value=[%u], rmtCKE=[%u:%04x], value=[%u --> %u].",
                rankId_, dieId_, channelId_, rmtCcu.first, rmtCcu.second, locCKEId_, locCKEMask_, locCKE, rmtCKEId_, locCKEMask_, newRmtCke);

    // 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void SyncCkeExecutor::Run() {
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "SyncCKE");
}

std::string SyncCkeExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Sync LocCKE[%u:%04x] To rmtCKE[%u:%04x] Use Channel[%u], Set "
                              "CKE[%u:%04x], clearType[%u]",
        waitCKEId_,
        waitCKEMask_,
        locCKEId_,
        locCKEMask_,
        rmtCKEId_,
        locCKEMask_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_);
}