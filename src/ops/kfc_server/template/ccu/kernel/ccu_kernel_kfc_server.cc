/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_kfc_server.h"
#include "const_val.h"

namespace ops_hccl {

using namespace hcomm;
using namespace std;

const string OP_SELECTOR_LABEL = "OpSelector";
const uint32_t HBM_PARAM_IDX_0 = 0;
const uint32_t HBM_PARAM_IDX_1 = 1;
const uint32_t HBM_PARAM_IDX_2 = 2;
const uint32_t HBM_PARAM_IDX_3 = 3;

const uint32_t SINGLE_DIE = 1;
const uint32_t DOUBLE_DIE = 2;
const uint32_t DIE0_ID = 0;
const uint32_t DIE1_ID = 1;
const string DIE1_START_SIG = "Die1StartSig";
const string DIE1_END_SIG = "Die1EndSig";

CcuKernelKfcServer::CcuKernelKfcServer(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgKfcServer *kernelArg = dynamic_cast<const CcuKernelArgKfcServer *>(&arg);
    rankId_ = kernelArg->rankId_;
    rankSize_ = kernelArg->dimSize_;
    channels_ = kernelArg->channels;
    loadFromMem_ = kernelArg->loadFromMem_;
    algoTemplateInfo_ = kernelArg->opParam_.algoTemplateInfo;
    HCCL_INFO("[CcuKernelKfcServer] rankId_ = %d, rankSize_ = %d, loadFromMem_ = %d", rankId_, rankSize_, loadFromMem_);
}

HcclResult CcuKernelKfcServer::InitResource()
{
    HCCL_INFO("[CcuKernelKfcServer] InitResource!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelKfcServer::Algorithm()
{
    GenOpSelector();
    GenCircularQueue();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelKfcServer::GenOpSelector()
{
    if (algoTemplateInfo_.empty()) {
        HCCL_ERROR("[CcuKernelKfcServer] GenOpSelector Failed: Empty AlgoTemplateInfo");
        return;
    }

    {
        std::string funcName = OP_SELECTOR_LABEL + "_" + std::to_string(GetDieId()) + "_" + std::to_string(missionIndex);
        CcuRep::FuncBlock selectorFunc(this, funcName);

        CcuRep::Variable opCode = CreateVariable();
        selectorFunc.DefineInArg(opCode);

        CcuRep::Variable opAddr = CreateVariable();
        selectorFunc.DefineOutArg(opAddr);
        opAddr = INVALID_U64;

        for (auto entry : algoTemplateInfo_) {
            CCU_IF(opCode == entry.first)
            {
                opAddr = entry.second;
            }
        }
    }
}

void CcuKernelKfcServer::GenCircularQueue()
{
    CcuRep::Variable token = CreateVariable();
    Load(token);

    CcuRep::Variable opAddr = CreateVariable();

    CcuRep::Variable repeatCond = CreateVariable();
    repeatCond = 0;

    CcuRep::Variable turnStartSig = CreateVariable();
    turnStartSig = 0;
    CcuRep::Variable turnEndSig = CreateVariable();
    turnEndSig = 1;

    CcuRep::Variable waitStartAddr = CreateVariable();
    waitStartAddr = waitAddr_;
    CcuRep::Variable recordStartAddr = CreateVariable();
    recordStartAddr = recordAddr_;
    CcuRep::Variable paramStartAddr = CreateVariable();
    paramStartAddr = paramAddr_;
    CcuRep::Variable waitAddr = CreateVariable();
    waitAddr = waitAddr_;
    CcuRep::Variable recordAddr = CreateVariable();
    recordAddr = recordAddr_;
    CcuRep::Variable paramAddr = CreateVariable();
    paramAddr = paramAddr_;

    CcuRep::Variable ckeSize = CreateVariable();
    ckeSize = CCU_ONE_PARAM_SIZE;
    CcuRep::Variable paramSize = CreateVariable();
    paramSize = CCU_PARAM_NUM_MAX * CCU_ONE_PARAM_SIZE;

    CcuRep::Variable queueIdx = CreateVariable();
    queueIdx = 0;
    CcuRep::Variable queueEnd = CreateVariable();
    queueEnd = CCU_TASK_NUM_MAX;
    CcuRep::Variable one = CreateVariable();
    one = 1;

    array<CcuRep::Variable, CCU_PARAM_NUM_PER_DIE> param;
    for (uint32_t i = 0; i < CCU_PARAM_NUM_PER_DIE; ++i) {
        param[i] = CreateContinuousVariable();
    }

    CCU_WHILE(repeatCond == 0)
    {
        if (waitAddr_ > (UINT64_MAX - (CCU_TASK_NUM_MAX - 1) * CCU_ONE_PARAM_SIZE)
            || recordAddr_ > (UINT64_MAX - (CCU_TASK_NUM_MAX - 1) * CCU_ONE_PARAM_SIZE)
            || paramAddr_ > (UINT64_MAX - (CCU_TASK_NUM_MAX - 1) * CCU_PARAM_NUM_MAX * CCU_ONE_PARAM_SIZE)) {
            HCCL_ERROR("[CcuKernelKfcServer] GenCircularQueue Failed: integer overflow occurs");
            return;
        }

        WaitTurnStartSig(waitAddr, turnStartSig);

        LoadFuncParamFromMemory(paramAddr, param);

        MissionPreSync(param[HBM_PARAM_IDX_0]);

        CCU_IF(param[HBM_PARAM_IDX_0] == INVALID_U64)
        {
            CCU_BREAK;
        }

        std::string funcName = OP_SELECTOR_LABEL + "_" + std::to_string(GetDieId()) + "_" + std::to_string(missionIndex);
        auto selectFunc = Func(funcName);
        selectFunc.SetInArg(param[HBM_PARAM_IDX_0]);
        selectFunc.SetOutArg(opAddr);
        selectFunc.AppendToContext();

        CCU_IF(opAddr == INVALID_U64)
        {
            CCU_BREAK;
        }

        auto opFunc = Func(opAddr);
        opFunc.SetInArg(param[HBM_PARAM_IDX_1]);
        opFunc.SetInArg(param[HBM_PARAM_IDX_2]);
        opFunc.SetInArg(token);
        for (uint32_t i = HBM_PARAM_IDX_3; i < CCU_PARAM_NUM_PER_DIE; ++i) {
            opFunc.SetInArg(param[i]);
        }
        opFunc.AppendToContext();

        MissionPostSync();

        SetTurnEndSig(recordAddr, turnEndSig);
        waitAddr += ckeSize;
        recordAddr += ckeSize;
        paramAddr += paramSize;
        queueIdx += one;
        CCU_IF (queueIdx == static_cast<u64>(CCU_TASK_NUM_MAX)) {
            waitAddr = waitStartAddr;
            recordAddr = recordStartAddr;
            paramAddr = paramStartAddr;
            queueIdx = 0;
        }
    }
}

void CcuKernelKfcServer::WaitTurnStartSig(const CcuRep::Variable &hbmSigAddr, CcuRep::Variable &turnStartSig)
{
    if (dieNum_ == SINGLE_DIE) {
        CCU_WHILE(turnStartSig != 1)
        {
            LoadVariable(hbmSigAddr, turnStartSig);
        }
        turnStartSig = 0;
        StoreVariable(turnStartSig, hbmSigAddr);
    } else {
        if (GetDieId() == DIE0_ID) {
            CCU_WHILE(turnStartSig != 1)
            {
                LoadVariable(hbmSigAddr, turnStartSig);
            }
            turnStartSig = 0;
            StoreVariable(turnStartSig, hbmSigAddr);
            LocalCtxPost(importDieSig, 1);
        } else if (GetDieId() == DIE1_ID) {
            LocalWait(exportDieSig, 1);
        }
    }
}

void CcuKernelKfcServer::SetTurnEndSig(const CcuRep::Variable &hbmSigAddr, const CcuRep::Variable &turnEndSig)
{
    if (dieNum_ == SINGLE_DIE) {
        StoreVariable(turnEndSig, hbmSigAddr);
    } else {
        if (GetDieId() == DIE0_ID) {
            LocalWait(exportDieSig, 1);
            StoreVariable(turnEndSig, hbmSigAddr);
        } else if (GetDieId() == DIE1_ID) {
            LocalCtxPost(importDieSig, 1);
        }
    }
}

void CcuKernelKfcServer::LoadFuncParamFromMemory(CcuRep::Variable &paramAddr, array<CcuRep::Variable, CCU_PARAM_NUM_PER_DIE> &param)
{
    CcuRep::Variable doubleDie = CreateVariable();
    doubleDie = CCU_PARAM_NUM_PER_DIE * CCU_ONE_PARAM_SIZE;
    CcuRep::Variable addr = CreateVariable();
    addr = paramAddr;
    if (dieNum_ == DOUBLE_DIE && GetDieId() == DIE1_ID) {
        addr += doubleDie;
    }
    LoadVariable(addr, param[0], CCU_PARAM_NUM_PER_DIE);
}

void CcuKernelKfcServer::MissionPreSync(CcuRep::Variable &func)
{
    if (missionNum == 1) {
        return;
    }
    if (missionIndex == 0) {
        for (uint32_t i = 0; i < missionNum - 1; ++i) {
            LocalCtxPostVar(func, importMissionVar[i], importMissionSig[i]);
        }
    } else {
        LocalWait(exportMissoinSig[0]);
        func = exportMissionVar[0];
    }
}

void CcuKernelKfcServer::MissionPostSync()
{
    if (missionNum == 1) {
        return;
    }
    if (missionIndex == 0) {
        for (uint32_t i = 0; i < missionNum - 1; ++i) {
            LocalWait(exportMissoinSig[i]);
        }
    } else {
        LocalCtxPost(importMissionSig[0]);
    }
}

std::vector<uint64_t> CcuKernelKfcServer::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgKfcServer *taskArg = dynamic_cast<const CcuTaskArgKfcServer *>(&arg);
    return {taskArg->inputAddr_, taskArg->outputAddr_};
}

}