/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans loc mem to rmt mem
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransLocMemToRmtMemExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransLocMemToRmtMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTORMTMEM_CODE, TransLocMemToRmtMemExecutor)

void TransLocMemToRmtMemExecutor::Parser()
{
    rmtGSAId_       = instr_.v1.transLocMemToRmtMem.rmtGSAId;
    rmtXnId_        = instr_.v1.transLocMemToRmtMem.rmtXnId;
    locGSAId_       = instr_.v1.transLocMemToRmtMem.locGSAId;
    locXnId_        = instr_.v1.transLocMemToRmtMem.locXnId;
    lengthXnId_     = instr_.v1.transLocMemToRmtMem.lengthXnId;
    channelId_      = instr_.v1.transLocMemToRmtMem.channelId;
    reduceDataType_ = instr_.v1.transLocMemToRmtMem.reduceDataType;
    reduceOpCode_   = instr_.v1.transLocMemToRmtMem.reduceOpCode;
    clearType_      = instr_.v1.transLocMemToRmtMem.clearType;
    lengthEn_       = instr_.v1.transLocMemToRmtMem.lengthEn;
    reduceEn_       = instr_.v1.transLocMemToRmtMem.reduceEn;
    setCKEId_       = instr_.v1.transLocMemToRmtMem.setCKEId;
    setCKEMask_     = instr_.v1.transLocMemToRmtMem.setCKEMask;
    waitCKEId_      = instr_.v1.transLocMemToRmtMem.waitCKEId;
    waitCKEMask_    = instr_.v1.transLocMemToRmtMem.waitCKEMask;
}

// 本端mem的数据搬运到对端mem中
void TransLocMemToRmtMemExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.要搬运的本端内存地址及数据长度
    uint64_t rmtAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, rmtGSAId_);
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        uint64_t offset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        rmtAddr += offset;
        locAddr += offset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransLocMemToRmtMemExecutor][Process] locCcu[%d:%d], Get gsa addr offset = [%x], cke offset = [%u]", rankId_, dieId_, offset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransLocMemToRmtMemExecutor][Process] locCcu[%d:%d] Trans data from locGSAId_[%u] locAddr[%llx] to rmtGSAId_[%u] rmtAddr[%llx], "
               "with lengthXnId[%u] transLength[%lu].", rankId_, dieId_, locGSAId_, locAddr, rmtGSAId_, rmtAddr, lengthXnId_, transLength_);
    ccuResMgr.TransMemToMem(reinterpret_cast<void *>(locAddr), reinterpret_cast<void *>(rmtAddr), transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMemToRmtMemExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMemToRmtMem");
}

std::string TransLocMemToRmtMemExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Trans LocMem[%u:%u] To RmtMem[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], Set "
                              "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        locGSAId_,
        locXnId_,
        rmtGSAId_,
        rmtXnId_,
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