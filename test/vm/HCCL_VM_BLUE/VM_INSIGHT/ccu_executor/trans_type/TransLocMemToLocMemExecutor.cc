/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans loc mem to loc mem
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "TransLocMemToLocMemExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册TransLocMemToLocMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTOLOCMEM_CODE, TransLocMemToLocMemExecutor)

void TransLocMemToLocMemExecutor::Parser()
{
    dstGSAId_    = instr_.v1.transLocMemToLocMem.dstGSAId;
    dstXnId_     = instr_.v1.transLocMemToLocMem.dstXnId;
    srcGSAId_    = instr_.v1.transLocMemToLocMem.srcGSAId;
    srcXnId_     = instr_.v1.transLocMemToLocMem.srcXnId;
    lengthXnId_  = instr_.v1.transLocMemToLocMem.lengthXnId;
    channelId_   = instr_.v1.transLocMemToLocMem.channelId;
    clearType_   = instr_.v1.transLocMemToLocMem.clearType;
    lengthEn_    = instr_.v1.transLocMemToLocMem.lengthEn;
    setCKEId_    = instr_.v1.transLocMemToLocMem.setCKEId;
    setCKEMask_  = instr_.v1.transLocMemToLocMem.setCKEMask;
    waitCKEId_   = instr_.v1.transLocMemToLocMem.waitCKEId;
    waitCKEMask_ = instr_.v1.transLocMemToLocMem.waitCKEMask;
}

// 本端源mem的数据搬运到本端目的mem中
void TransLocMemToLocMemExecutor::Process(CcuResouceManager &ccuResMgr)
{
    // 1.要搬运的本端内存地址及数据长度
    uint64_t srcLocAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, srcGSAId_);
    uint64_t dstLocAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, dstGSAId_);
    transLength_ = (lengthEn_ == 0) ? BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        uint64_t gsaOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        uint16_t ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        srcLocAddr += gsaOffset;
        dstLocAddr += gsaOffset;
        setCKEId_ += ckeOffset;
        HCCL_DEBUG("[TransLocMemToLocMemExecutor][Process] locCcu[%d:%d], Get gsa addr offset = [%x], cke id offset = [%u]", rankId_, dieId_, gsaOffset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_DEBUG("[TransLocMemToLocMemExecutor][Process] locCcu[%d:%d] Trans data from srcLocGSAId_[%u] srcLocAddr[%llx] to dstLocGSAId_[%u] "
               "dstLocAddr[%llx], with lengthXnId[%u] transLength[%lu].",
        rankId_, dieId_, srcGSAId_, srcLocAddr, dstGSAId_, dstLocAddr, lengthXnId_, transLength_);
    ccuResMgr.TransMemToMem(reinterpret_cast<void *>(srcLocAddr), reinterpret_cast<void *>(dstLocAddr), transLength_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMemToLocMemExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMemToLocMem");
}

std::string TransLocMemToLocMemExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMem[%u:%u] With "
                              "LengthXn[%u] Use Channel[%u], Set "
                              "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
        waitCKEId_,
        waitCKEMask_,
        srcGSAId_,
        srcXnId_,
        dstGSAId_,
        dstXnId_,
        lengthXnId_,
        channelId_,
        setCKEId_,
        setCKEMask_,
        clearType_,
        lengthEn_);
}