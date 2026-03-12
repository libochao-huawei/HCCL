/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans locms to rmtms
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransLocMSToRmtMSExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransLocMSToRmtMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTORMTMS_CODE, TransLocMSToRmtMSExecutor)

void TransLocMSToRmtMSExecutor::Parser()
{
    rmtMSId_       = instr_.v1.transLocMSToRmtMS.rmtMSId & 0x7FFF;
    rmtDieId_      = instr_.v1.transLocMSToRmtMS.rmtMSId >> 15;
    locMSId_       = instr_.v1.transLocMSToRmtMS.locMSId & 0x7FFF;
    locDieId_      = instr_.v1.transLocMSToRmtMS.locMSId >> 15;
    lengthXnId_    = instr_.v1.transLocMSToRmtMS.lengthXnId;
    channelId_     = instr_.v1.transLocMSToRmtMS.channelId;
    setRmtCKEId_   = instr_.v1.transLocMSToRmtMS.setRmtCKEId;
    setRmtCKEMask_ = instr_.v1.transLocMSToRmtMS.setRmtCKEMask;
    clearType_     = instr_.v1.transLocMSToRmtMS.clearType;
    lengthEn_      = instr_.v1.transLocMSToRmtMS.lengthEn;
    setCKEId_      = instr_.v1.transLocMSToRmtMS.setCKEId;
    setCKEMask_    = instr_.v1.transLocMSToRmtMS.setCKEMask;
    waitCKEId_     = instr_.v1.transLocMSToRmtMS.waitCKEId;
    waitCKEMask_   = instr_.v1.transLocMSToRmtMS.waitCKEMask;
}

// 本端的源MS的数据搬运到对端的目的MS中
void TransLocMSToRmtMSExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(dieId_, channelId_);
    if (rmtCcu.second != rmtDieId_) {
        HCCL_ERROR("[TransLocMSToRmtMSExecutor][Process] dieId[%d] from channel is not same as rmtDieId[%d]. curCcu[%d:%d], rmtCcu[%d:%d]",
            rmtCcu.second, rmtDieId_, rankId_, dieId_, rmtCcu.first, rmtCcu.second);
        return;
    }
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        rmtMSId_ += msOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransLocMSToRmtMSExecutor][Process] curCcu=[%d:%d], Get ms offset = [%u], cke offset = [%u]", rankId_, dieId_, msOffset, ckeOffset);
    }
    // 3.要搬运的本端内存地址及数据长度
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 4.搬运动作
    HCCL_DEBUG("[TransLocMSToRmtMSExecutor][Process] ccuId=[%d:%d-%d:%d] Trans data from locSrcMsId[%u] to rmtDstMsId[%u], "
               "with lengthXnId[%u] transLength[%u].", rankId_, dieId_, rmtCcu.first, rmtCcu.second, locMSId_, rmtMSId_, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMS(rankId_, dieId_, rmtCcu.first, rmtCcu.second, locMSId_, rmtMSId_, transLength_);
    // 5.设置对端的cke
    SetRmtCKESignal(ccuResMgr, rmtCcu.first, rmtCcu.second, setRmtCKEId_, setRmtCKEMask_);
    // 6.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToRmtMSExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransLocMsToRmtMs");
}

std::string TransLocMSToRmtMSExecutor::Describe()
{
    return Hccl::StringFormat("ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans LocMS[%u:%u] To RmtMS[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], Set "
                              "RmtCKE[%u:%04x], Set CKE[%u:%04x], clearType[%u], lengthEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        locMSId_ / 0x8000,
        locMSId_ % 0x8000,
        rmtMSId_ / 0x8000,
        rmtMSId_ % 0x8000,
        lengthXnId_,
        channelId_,
        setRmtCKEId_,
        setRmtCKEMask_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}