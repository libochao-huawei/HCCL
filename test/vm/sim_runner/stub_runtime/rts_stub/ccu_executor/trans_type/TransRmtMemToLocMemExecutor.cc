/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans rmt mem to loc mem
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransRmtMemToLocMemExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransRmtMemToLocMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMEMTOLOCMEM_CODE, TransRmtMemToLocMemExecutor)

void TransRmtMemToLocMemExecutor::Parser()
{
    locGSAId_       = instr_.v1.transRmtMemToLocMem.locGSAId;
    locXnId_        = instr_.v1.transRmtMemToLocMem.locXnId;
    rmtGSAId_       = instr_.v1.transRmtMemToLocMem.rmtGSAId;
    rmtXnId_        = instr_.v1.transRmtMemToLocMem.rmtXnId;
    lengthXnId_     = instr_.v1.transRmtMemToLocMem.lengthXnId;
    channelId_      = instr_.v1.transRmtMemToLocMem.channelId;
    reduceDataType_ = instr_.v1.transRmtMemToLocMem.reduceDataType;
    reduceOpCode_   = instr_.v1.transRmtMemToLocMem.reduceOpCode;
    clearType_      = instr_.v1.transRmtMemToLocMem.clearType;
    lengthEn_       = instr_.v1.transRmtMemToLocMem.lengthEn;
    reduceEn_       = instr_.v1.transRmtMemToLocMem.reduceEn;
    setCKEId_       = instr_.v1.transRmtMemToLocMem.setCKEId;
    setCKEMask_     = instr_.v1.transRmtMemToLocMem.setCKEMask;
    waitCKEId_      = instr_.v1.transRmtMemToLocMem.waitCKEId;
    waitCKEMask_    = instr_.v1.transRmtMemToLocMem.waitCKEMask;
}

// 对端Mem的数据搬运到本端的Mem中
void TransRmtMemToLocMemExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.要搬运的远端内存地址及数据长度
    uint64_t rmtAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, rmtGSAId_);
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto offset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        rmtAddr += offset;
        locAddr += offset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransRmtMemToLocMemExecutor][Process] locCcu[%d:%d], Get gsa addr offset = [%x], cke offset = [%x]", rankId_, dieId_, offset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransRmtMemToLocMemExecutor][Process] locCcu[%d:%d] Trans data from rmtGSAId[%u] rmtAddr[%llx] to locGSAId[%u] locAddr[%llx], "
               "with lengthXnId[%u] transLength[%lu].", rankId_, dieId_, rmtGSAId_, rmtAddr, locGSAId_, locAddr, lengthXnId_, transLength_);
    ccuResMgr.TransMemToMem(reinterpret_cast<void *>(rmtAddr), reinterpret_cast<void *>(locAddr), transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMemToLocMemExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransRmtMemToLocMem");
}

std::string TransRmtMemToLocMemExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMem[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], Set "
                              "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        rmtGSAId_,
        rmtXnId_,
        locGSAId_,
        locXnId_,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_,
        reduceDataType_,
        reduceOpCode_,
        reduceEn_);
}