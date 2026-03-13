/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans rmt ms to loc ms
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransRmtMSToLocMSExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransRmtMSToLocMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMSTOLOCMS_CODE, TransRmtMSToLocMSExecutor)

void TransRmtMSToLocMSExecutor::Parser()
{
    locMSId_     = instr_.v1.transRmtMSToLocMS.locMSId & 0x7FFF;
    locDieId_    = instr_.v1.transRmtMSToLocMS.locMSId >> 15;
    rmtMSId_     = instr_.v1.transRmtMSToLocMS.rmtMSId & 0x7FFF;
    rmtDieId_    = instr_.v1.transRmtMSToLocMS.rmtMSId >> 15;
    lengthXnId_  = instr_.v1.transRmtMSToLocMS.lengthXnId;
    channelId_   = instr_.v1.transRmtMSToLocMS.channelId;
    clearType_   = instr_.v1.transRmtMSToLocMS.clearType;
    lengthEn_    = instr_.v1.transRmtMSToLocMS.lengthEn;
    setCKEId_    = instr_.v1.transRmtMSToLocMS.setCKEId;
    setCKEMask_  = instr_.v1.transRmtMSToLocMS.setCKEMask;
    waitCKEId_   = instr_.v1.transRmtMSToLocMS.waitCKEId;
    waitCKEMask_ = instr_.v1.transRmtMSToLocMS.waitCKEMask;
}

// 对端的源MS的数据搬运到本端的目的MS中
void TransRmtMSToLocMSExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(dieId_, channelId_);
    if (rmtCcu.second != rmtDieId_) {
        HCCL_ERROR("[TransRmtMSToLocMSExecutor][Process] dieId[%d] from channel is not same as rmtDieId[%d]. curCcu[%d:%d], rmtCcu[%d:%d]",
            rmtCcu.second, rmtDieId_, rankId_, dieId_, rmtCcu.first, rmtCcu.second);
        return;
    }
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        locMSId_ += msOffset;
        rmtMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransRmtMSToLocMSExecutor][Process] ccuId=[%d:%d], Get ms offset = [%u], cke offset = [%u]", rankId_, dieId_, msOffset, ckeOffset);
    }
    // 3.要搬运的本端内存地址及数据长度
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 4.搬运动作
    HCCL_DEBUG("[TransRmtMSToLocMSExecutor][Process] ccuId=[%d:%d-%d:%d] Trans data from rmtMsId[%u] to locMsId[%u], "
               "with lengthXnId[%u] transLength[%u].", rankId_, dieId_, rmtCcu.first, rmtCcu.second, rmtMSId_, locMSId_, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMS(rmtCcu.first, rmtCcu.second, rankId_, dieId_, rmtMSId_, locMSId_, transLength_);
    // 5.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMSToLocMSExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransRmtMsToLocMs");
}

std::string TransRmtMSToLocMSExecutor::Describe()
{
    return Hccl::StringFormat("ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMS[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], "
                              "Set CKE[%u:%04x], "
                              "clearType[%u], lengthEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        rmtMSId_ / 0x8000,
        rmtMSId_ % 0x8000,
        locMSId_ / 0x8000,
        locMSId_ % 0x8000,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}