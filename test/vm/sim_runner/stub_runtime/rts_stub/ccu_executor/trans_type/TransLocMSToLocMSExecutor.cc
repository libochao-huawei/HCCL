/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans locms to locms
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransLocMSToLocMSExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransLocMSToLocMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTOLOCMS_CODE, TransLocMSToLocMSExecutor)

void TransLocMSToLocMSExecutor::Parser()
{
    dstMSId_     = instr_.v1.transLocMSToLocMS.dstMSId & 0x7FFF; // 0x7FFF取最低15bit的值
    dstDieId_    = instr_.v1.transLocMSToLocMS.dstMSId >> 15; // 取bit15的值
    srcMSId_     = instr_.v1.transLocMSToLocMS.srcMSId & 0x7FFF; // 0x7FFF取最低15bit的值
    srcDieId_    = instr_.v1.transLocMSToLocMS.srcMSId >> 15; // 取bit15的值
    lengthXnId_  = instr_.v1.transLocMSToLocMS.lengthXnId;
    channelId_   = instr_.v1.transLocMSToLocMS.channelId;
    clearType_   = instr_.v1.transLocMSToLocMS.clearType;
    lengthEn_    = instr_.v1.transLocMSToLocMS.lengthEn;
    setCKEId_    = instr_.v1.transLocMSToLocMS.setCKEId;
    setCKEMask_  = instr_.v1.transLocMSToLocMS.setCKEMask;
    waitCKEId_   = instr_.v1.transLocMSToLocMS.waitCKEId;
    waitCKEMask_ = instr_.v1.transLocMSToLocMS.waitCKEMask;
}

// 本端的源MS的数据搬运到本端的目的MS中
void TransLocMSToLocMSExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.要搬运的本端内存地址及数据长度
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        srcMSId_ += msOffset;
        dstMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransLocMSToLocMSExecutor][Process] locCcu[%d:%d], Get ms offset = [%u], cke offset = [%u]", rankId_, dieId_, msOffset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransLocMSToLocMSExecutor][Process] locCcu[%d:%d] Trans data from locSrcMsId[%u] to locDstMsId[%u], "
               "with lengthXnId[%u] transLength[%u].", rankId_, dieId_, srcMSId_, dstMSId_, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMS(rankId_, dieId_, rankId_, dieId_, srcMSId_, dstMSId_, transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToLocMSExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransLocMsToLocMs");
}

std::string TransLocMSToLocMSExecutor::Describe()
{
    return Hccl::StringFormat("ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMS[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], "
                              "Set CKE[%u:%04x], "
                              "clearType[%u], lengthEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        srcMSId_ / 0x8000,
        srcMSId_ % 0x8000,
        dstMSId_ / 0x8000,
        dstMSId_ % 0x8000,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}