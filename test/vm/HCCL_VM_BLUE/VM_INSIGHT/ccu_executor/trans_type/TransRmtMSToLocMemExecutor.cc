/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans rmt ms to loc mem
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransRmtMSToLocMemExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransRmtMSToLocMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMSTOLOCMEM_CODE, TransRmtMSToLocMemExecutor)

void TransRmtMSToLocMemExecutor::Parser()
{
    locGSAId_    = instr_.v1.transRmtMSToLocMem.locGSAId;
    locXnId_     = instr_.v1.transRmtMSToLocMem.locXnId;
    rmtMSId_     = instr_.v1.transRmtMSToLocMem.rmtMSId & 0x7FFF;
    rmtDieId_    = instr_.v1.transRmtMSToLocMem.rmtMSId >> 15;
    lengthXnId_  = instr_.v1.transRmtMSToLocMem.lengthXnId;
    channelId_   = instr_.v1.transRmtMSToLocMem.channelId;
    clearType_   = instr_.v1.transRmtMSToLocMem.clearType;
    lengthEn_    = instr_.v1.transRmtMSToLocMem.lengthEn;
    setCKEId_    = instr_.v1.transRmtMSToLocMem.setCKEId;
    setCKEMask_  = instr_.v1.transRmtMSToLocMem.setCKEMask;
    waitCKEId_   = instr_.v1.transRmtMSToLocMem.waitCKEId;
    waitCKEMask_ = instr_.v1.transRmtMSToLocMem.waitCKEMask;
}

// 对端的源MS的数据搬运到本端的目的MEM中
void TransRmtMSToLocMemExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(dieId_, channelId_);
    if (rmtCcu.second != rmtDieId_) {
        HCCL_ERROR("[TransRmtMSToLocMemExecutor][Process] dieId[%d] from channel is not same as rmtDieId[%d]. curCcu[%d:%d], rmtCcu[%d:%d]",
            rmtCcu.second, rmtDieId_, rankId_, dieId_, rmtCcu.first, rmtCcu.second);
        return;
    }
    // 2.要搬运的远端内存地址及数据长度
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    // 3.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        locAddr  += addrOffset;
        rmtMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransRmtMSToLocMemExecutor][Process] ccuId=[%d:%d], Get gsa addr offset = [%x], ms offset = [%u], cke offset = [%u]", rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    // 4.要搬运的本端内存地址及数据长度
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 5.搬运动作
    HCCL_DEBUG("[TransRmtMSToLocMemExecutor][Process] ccuId=[%d:%d-%d:%d] Trans data from rmtMsId[%u] to locGSAId[%u] locAddr[%llx], "
               "with lengthXnId[%u] transLength[%u].", rankId_, dieId_, rmtCcu.first, rmtCcu.second, rmtMSId_, locGSAId_, locAddr, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMem(rmtCcu.first, rmtCcu.second, rmtMSId_, reinterpret_cast<void *>(locAddr), transLength_);
    // 6.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMSToLocMemExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransRmtMsToLocMem");
}

std::string TransRmtMSToLocMemExecutor::Describe()
{
    return Hccl::StringFormat("ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMem[%u:%u] "
                              "With LengthXn[%u] Use Channel[%u], Set "
                              "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        rmtMSId_ / 0x8000,
        rmtMSId_ % 0x8000,
        locGSAId_,
        locXnId_,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}