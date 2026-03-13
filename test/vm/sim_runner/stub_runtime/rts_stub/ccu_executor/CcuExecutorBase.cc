/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor base
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "CcuExecutorBase.h"
#include "CcuResourceManager.h"

using namespace std;
using namespace Hccl::CcuRep;

std::string CcuExecutorBase::ParseMSList()
{
    // 待实现，检查sqe类型
    uint16_t msId[CCU_REDUCE_MAX_MS];
    uint16_t count = instr_.v1.add.count;
    for (uint16_t index = 0; index < CCU_REDUCE_MAX_MS; index++) {
        msId[index] = instr_.v1.add.msId[index];
    }

    std::string res = "MS[";
    for (uint16_t i = 0; i < count + 2; i++) { // 循环范围 0~count + 2
        if (i == count + 1) {
            res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + "]";
        } else {
            res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + ", ";
        }
    }
    return res;
}

void CcuExecutorBase::SetCkeSignal(CcuResouceManager &ccuResMgr, uint16_t setCKEId, uint16_t setCKEMask)
{
    uint16_t setCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, setCKEId);
    uint16_t newSetCKE = (setCKE & (~setCKEMask)) | setCKEMask;
    ccuResMgr.UpdateCkeValue(rankId_, dieId_, setCKEId, newSetCKE);
    HCCL_DEBUG("[CcuExecutorBase][SetCkeSignal] success, locCcu[%d:%d], SetCKE[%u:%04x], value[%u --> %u]",
        rankId_,
        dieId_,
        setCKEId,
        setCKEMask,
        setCKE,
        newSetCKE);
}

void CcuExecutorBase::SetRmtCKESignal(CcuResouceManager &ccuResMgr, int rmtRank, int rmtDie, uint16_t setRmtCKEId, uint16_t setRmtCKEMask)
{
    uint16_t rmtCKE = ccuResMgr.GetCkeValue(rmtRank, rmtDie, setRmtCKEId);
    uint16_t newRmtCKE = (rmtCKE & (~setRmtCKEMask)) | setRmtCKEMask;
    ccuResMgr.UpdateCkeValue(rmtRank, rmtDie, setRmtCKEId, newRmtCKE);
    HCCL_DEBUG("[CcuExecutorBase][SetRmtCKESignal] success, ccu[%d:%d --> %d:%d], SetRmtCKE[%u:%04x], value[%u --> %u]",
        rankId_,
        dieId_,
        rmtRank,
        rmtDie,
        setRmtCKEId,
        setRmtCKEMask,
        rmtCKE,
        newRmtCKE);
}

void CcuExecutorBase::ClearCkeSignal(CcuResouceManager &ccuResMgr, uint16_t clearCKEId, uint16_t clearMask)
{
    // 判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        clearCKEId += ckeOffset;
        HCCL_DEBUG("[CcuExecutorBase][ClearCkeSignal] locCcu[%d:%d], Get cke id offset = [%x]", rankId_, dieId_, ckeOffset);
    }
    uint16_t setCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, clearCKEId);
    uint16_t newSetCKE = setCKE & (~clearMask);
    ccuResMgr.UpdateCkeValue(rankId_, dieId_, clearCKEId, newSetCKE);
    HCCL_DEBUG("[CcuExecutorBase][ClearCkeSignal] success, locCcu[%d:%d], ClearCKE[%u:%04x], value[%u --> %u]",
        rankId_,
        dieId_,
        clearCKEId,
        clearMask,
        setCKE,
        newSetCKE);
}

void CcuExecutorBase::WaitCkeProcess(uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, const std::string &instrName)
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    // 判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto ckeOffset   = ccuSimulator_->GetLoopCKEOffset();
        waitCKEId += ckeOffset;
        HCCL_DEBUG("[CcuExecutorBase][ClearCkeSignal] instr[%s], locCcu[%d:%d], Get cke id offset = [%x]", instrName.c_str(), rankId_, dieId_, ckeOffset);
    }
    auto waitCKE = ccuResMgr.GetCkeValue(rankId_, dieId_, waitCKEId);
    if (waitCKEMask != 0) {
        if ((waitCKE & waitCKEMask) == waitCKEMask) {
            // 指令具体操作
            Process(ccuResMgr);
        } else {
            HCCL_DEBUG("[CcuExecutorBase][WaitCkeProcess] instr[%s], blocking...ccuId=[%d:%d], waitCKE[%u:%04x], expect:[%04x], actual:[%04x], waitCKE:[%04x]",
                instrName.c_str(), rankId_, dieId_, waitCKEId, waitCKEMask, waitCKEMask, (waitCKE & waitCKEMask), waitCKE);
            ccuSimulator_->SetWaitCKEFlag(true);
            return;
        }
    } else {
        // 指令具体操作
        Process(ccuResMgr);
    }
    ccuSimulator_->SetWaitCKEFlag(false);
    // 执行完成后，清除本端cke
    if (clearType == 1) {
        ClearCkeSignal(ccuResMgr,  waitCKEId, waitCKEMask);
    }
}