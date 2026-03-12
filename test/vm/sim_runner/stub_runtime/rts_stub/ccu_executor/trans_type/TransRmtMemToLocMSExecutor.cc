/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans rmt mem to loc ms
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransRmtMemToLocMSExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransRmtMemToLocMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMEMTOLOCMS_CODE, TransRmtMemToLocMSExecutor)

void TransRmtMemToLocMSExecutor::Parser()
{
    locMSId_     = instr_.v1.transRmtMemToLocMS.locMSId & 0x7FFFU;
    locDieId_    = instr_.v1.transRmtMemToLocMS.locMSId >> 15U;
    rmtGSAId_    = instr_.v1.transRmtMemToLocMS.rmtGSAId;
    rmtXnId_     = instr_.v1.transRmtMemToLocMS.rmtXnId;
    lengthXnId_  = instr_.v1.transRmtMemToLocMS.lengthXnId;
    channelId_   = instr_.v1.transRmtMemToLocMS.channelId;
    clearType_   = instr_.v1.transRmtMemToLocMS.clearType;
    lengthEn_    = instr_.v1.transRmtMemToLocMS.lengthEn;
    setCKEId_    = instr_.v1.transRmtMemToLocMS.setCKEId;
    setCKEMask_  = instr_.v1.transRmtMemToLocMS.setCKEMask ;
    waitCKEId_   = instr_.v1.transRmtMemToLocMS.waitCKEId;
    waitCKEMask_ = instr_.v1.transRmtMemToLocMS.waitCKEMask;
}

// 本端MS的数据搬运到本端内存中
void TransRmtMemToLocMSExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.要搬运的远端内存地址及数据长度
    uint64_t rmtAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, rmtGSAId_);
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        rmtAddr  += addrOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransRmtMemToLocMSExecutor][Process] locCcu[%d:%d], Get gsa addr offset = [%x], ms offset = [%u], cke offset = [%u]", rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransRmtMemToLocMSExecutor][Process] locCcu[%d:%d] Trans data from rmtGSAId[%u] rmtAddr[%llx] to locMsId[%u], "
               "with lengthXnId[%u] transLength[%u].", rankId_, dieId_, rmtGSAId_, rmtAddr, locMSId_, lengthXnId_, transLength_);
    ccuResMgr.TransMemToMS(rankId_, dieId_, locMSId_, reinterpret_cast<void *>(rmtAddr), transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMemToLocMSExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "RmtMemToLocMS");
}

std::string TransRmtMemToLocMSExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMS[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], Set "
                              "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        rmtGSAId_,
        rmtXnId_,
        locMSId_ / 0x8000,
        locMSId_ % 0x8000,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}