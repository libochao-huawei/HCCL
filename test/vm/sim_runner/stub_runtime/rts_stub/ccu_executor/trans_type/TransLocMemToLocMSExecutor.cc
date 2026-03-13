/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans loc mem to loc ms
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransLocMemToLocMSExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransLocMemToLocMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTOLOCMS_CODE, TransLocMemToLocMSExecutor)

void TransLocMemToLocMSExecutor::Parser()
{
    locGSAId_    = instr_.v1.transLocMemToLocMS.locGSAId;
    locXnId_     = instr_.v1.transLocMemToLocMS.locXnId;
    locMSId_     = instr_.v1.transLocMemToLocMS.locMSId & 0x7fff; //  0x7fff取最低15位
    locDieId_    = instr_.v1.transLocMemToLocMS.locMSId >> 15; // 取bit15的值
    lengthXnId_  = instr_.v1.transLocMemToLocMS.lengthXnId;
    channelId_   = instr_.v1.transLocMemToLocMS.channelId;
    clearType_   = instr_.v1.transLocMemToLocMS.clearType;
    lengthEn_    = instr_.v1.transLocMemToLocMS.lengthEn;
    setCKEId_    = instr_.v1.transLocMemToLocMS.setCKEId;
    setCKEMask_  = instr_.v1.transLocMemToLocMS.setCKEMask;
    waitCKEId_   = instr_.v1.transLocMemToLocMS.waitCKEId;
    waitCKEMask_ = instr_.v1.transLocMemToLocMS.waitCKEMask;
}

// 本端MS的数据搬运到本端内存中
void TransLocMemToLocMSExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.要搬运的本端内存地址及数据长度
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        locAddr  += addrOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransLocMemToLocMSExecutor][Process] locCcu[%d:%d], Get gsa addr offset = [%x], ms offset = [%u], cke offset = [%u]", rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransLocMemToLocMSExecutor][Process] locCcu[%d:%d] Trans data from locGSAId_[%u] locAddr[%llx] to locMsId[%u], "
               "with lengthXnId[%u] transLength[%u], lengthEn_[%d].", rankId_, dieId_, locGSAId_, locAddr, locMSId_, lengthXnId_, transLength_, lengthEn_);
    ccuResMgr.TransMemToMS(rankId_, dieId_, locMSId_, reinterpret_cast<void *>(locAddr), transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMemToLocMSExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMemToLocMS");
}

std::string TransLocMemToLocMSExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
                        "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
                        waitCKEId_, waitCKEMask_, locGSAId_, locXnId_, locMSId_ / 0x8000, locMSId_ % 0x8000, lengthXnId_,
                        channelId_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
}