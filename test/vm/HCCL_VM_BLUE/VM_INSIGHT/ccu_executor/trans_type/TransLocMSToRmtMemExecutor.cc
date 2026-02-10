/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans loc ms to rmt mem
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransLocMSToRmtMemExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransLocMSToRmtMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTORMTMEM_CODE, TransLocMSToRmtMemExecutor)

void TransLocMSToRmtMemExecutor::Parser()
{
    rmtGSAId_    = instr_.v1.transLocMSToRmtMem.rmtGSAId;
    rmtXnId_     = instr_.v1.transLocMSToRmtMem.rmtXnId;
    locMSId_     = instr_.v1.transLocMSToRmtMem.locMSId & 0x7FFF;
    locDieId_    = instr_.v1.transLocMSToRmtMem.locMSId >> 15; // 取bit15的值
    lengthXnId_  = instr_.v1.transLocMSToRmtMem.lengthXnId;
    channelId_   = instr_.v1.transLocMSToRmtMem.channelId;
    clearType_   = instr_.v1.transLocMSToRmtMem.clearType;
    lengthEn_    = instr_.v1.transLocMSToRmtMem.lengthEn;
    setCKEId_    = instr_.v1.transLocMSToRmtMem.setCKEId;
    setCKEMask_  = instr_.v1.transLocMSToRmtMem.setCKEMask;
    waitCKEId_   = instr_.v1.transLocMSToRmtMem.waitCKEId;
    waitCKEMask_ = instr_.v1.transLocMSToRmtMem.waitCKEMask;
}

// 本端MS的数据搬运到远端端内存中
void TransLocMSToRmtMemExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.要搬运的远端内存地址及数据长度
    uint64_t rmtAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, rmtGSAId_);
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        rmtAddr  += addrOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransLocMSToRmtMemExecutor][Process] locCcu[%d:%d], Get gsa addr offset = [%x], ms offset = [%u], cke offset = [%u]", rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransLocMSToRmtMemExecutor][Process] locCcu[%d:%d] Trans data from locMsId[%u] to rmtGSAId_[%u] rmtAddr[%llx], "
               "with lengthXnId[%u] transLength[%u].", rankId_, dieId_, locMSId_, rmtGSAId_, rmtAddr, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMem(rankId_, dieId_, locMSId_, reinterpret_cast<void *>(rmtAddr), transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToRmtMemExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMSToRmtMem");
}

std::string TransLocMSToRmtMemExecutor::Describe()
{
    return Hccl::StringFormat("Wait CKE[%u:%04x], Trans LocMS[%u:%u] To RmtMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
                        "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
                        waitCKEId_, waitCKEMask_, locMSId_ / 0x8000, locMSId_ % 0x8000, rmtGSAId_, rmtXnId_, lengthXnId_,
                        channelId_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
}