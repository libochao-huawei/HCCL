/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans locms to locmem
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransLocMSToLocMemExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransLocMSToLocMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTOLOCMEM_CODE, TransLocMSToLocMemExecutor)

void TransLocMSToLocMemExecutor::Parser()
{
    locGSAId_    = instr_.v1.transLocMSToLocMem.locGSAId;
    locXnId_     = instr_.v1.transLocMSToLocMem.locXnId;
    locMSId_     = instr_.v1.transLocMSToLocMem.locMSId & 0x7FFF;
    locDieId_    = instr_.v1.transLocMSToLocMem.locMSId >> 15;
    lengthXnId_  = instr_.v1.transLocMSToLocMem.lengthXnId;
    channelId_   = instr_.v1.transLocMSToLocMem.channelId;
    clearType_   = instr_.v1.transLocMSToLocMem.clearType;
    lengthEn_    = instr_.v1.transLocMSToLocMem.lengthEn;
    setCKEId_    = instr_.v1.transLocMSToLocMem.setCKEId;
    setCKEMask_  = instr_.v1.transLocMSToLocMem.setCKEMask;
    waitCKEId_   = instr_.v1.transLocMSToLocMem.waitCKEId;
    waitCKEMask_ = instr_.v1.transLocMSToLocMem.waitCKEMask;
}

// 本端MS的数据搬运到本端内存中
void TransLocMSToLocMemExecutor::Process(CcuResouceManager &ccuResMgr)
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
        HCCL_DEBUG("[TransLocMSToLocMemExecutor][Process] locCcu[%d:%d], Get gsa addr offset = [%x], ms offset = [%u], "
                   "cke offset = [%u]", rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransLocMSToLocMemExecutor][Process] locCcu[%d:%d] Trans data from locMsId[%u] to locGSAId[%u] locAddr[%llx], "
               "with lengthXnId[%u] transLength[%u], lengthEn_[%d].", rankId_, dieId_, locMSId_, locGSAId_, locAddr, lengthXnId_, transLength_, lengthEn_);
    ccuResMgr.TransMSToMem(rankId_, dieId_, locMSId_, reinterpret_cast<void *>(locAddr), transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToLocMemExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMSToLocMem");
}

std::string TransLocMSToLocMemExecutor::Describe()
{
    return Hccl::StringFormat("ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
                        "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
                        waitCKEId_, waitCKEMask_, locMSId_ / 0x8000, locMSId_ % 0x8000, locGSAId_, locXnId_, lengthXnId_,
                        channelId_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
}