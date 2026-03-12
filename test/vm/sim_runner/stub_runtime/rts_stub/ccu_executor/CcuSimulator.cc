/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- jmp
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "CcuSimulator.h"
#include "CcuExecutorManager.h"
#include "CcuResourceManager.h"

using namespace std;
using namespace Hccl::CcuRep;

void CcuSimulator::InitLoopGroupInfo(uint16_t startLoopId, uint64_t offsetCfg, uint64_t repeatCfg)
{
    state_ =  CcuExecState::EXEC_LOOPGROUP_INSTR;
    loopGroupInfo_.startLoopId_   = startLoopId;
    loopGroupInfo_.loopNum_       = (repeatCfg >> 41) & 0x7F; // 0x7F: 取loop指令个数，即[47:41]位
    loopGroupInfo_.loopOffset_    = (repeatCfg >> 48) & 0x7F; // 0x7F: 取loop偏移，即[54:48]位
    loopGroupInfo_.loopExtendNum_ = (repeatCfg >> 55) & 0x7F; // 0x7F: 取loop展开次数，即[61:55]位
    HCCL_DEBUG("[CcuSimulator][InitLoopGroupInfo] locCcu[%d:%d], curInstrId_=[%u], loopNum=[%u], loopOffset=[%u], loopExtendNum=[%u]",
        rankId_, dieId_, curInstrId_, loopGroupInfo_.loopNum_, loopGroupInfo_.loopOffset_, loopGroupInfo_.loopExtendNum_);

    loopGroupInfo_.ckeOffset_ = offsetCfg & 0x3FF;               // 0x3FF: 取低[9:0]位
    loopGroupInfo_.msOffset_  = (offsetCfg >> 10) & 0x7FF;       // 0x7FF: 取低[20:10]位
    loopGroupInfo_.gsaOffset_ = (offsetCfg >> 21) & 0xFFFFFFFF;  // 0xFFFFFFFF: 取低[52:21]位
    HCCL_DEBUG("[CcuSimulator][InitLoopGroupInfo] locCcu[%d:%d], ckeOffset=[%u], msOffset=[%u], gsaOffset=[%u]",
        rankId_, dieId_, loopGroupInfo_.ckeOffset_, loopGroupInfo_.msOffset_, loopGroupInfo_.gsaOffset_);
}

void CcuSimulator::InitLoopInfo(uint16_t startInstrId, uint16_t endInstrId, uint16_t execCount, uint32_t addrStep)
{
    loopGroupInfo_.loopStatus_.loopStartInstrId = startInstrId;
    loopGroupInfo_.loopStatus_.loopCurInstrId   = startInstrId;
    loopGroupInfo_.loopStatus_.loopEndInstrId   = endInstrId;
    loopGroupInfo_.loopStatus_.loopExecCount    = execCount;
    loopGroupInfo_.loopStatus_.curLoopRound     = 0;
    loopGroupInfo_.loopStatus_.loopGsaIterStep  = addrStep;

    HCCL_DEBUG("[CcuSimulator][InitLoopInfo] locCcu[%d:%d], loopStartInstrId=[%u], loopCurInstrId=[%u], "
               "loopEndInstrId=[%u], loopExecCount=[%u], loopGsaIterStep=[%u]",
        rankId_, dieId_, startInstrId, startInstrId, endInstrId, execCount, addrStep);
}

void CcuSimulator::InitLoopGroupInfo(const LoopGroupInfo &loopGroupInfo)
{
    loopGroupInfo_ = loopGroupInfo;
}

uint64_t CcuSimulator::GetLoopGsaAddrOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    uint64_t addrOffset = loopGroupInfo_.gsaOffset_ * loopGroupInfo_.loopStatus_.loopExtendIndex +
                          loopGroupInfo_.loopStatus_.loopGsaIterStep * loopGroupInfo_.loopStatus_.curLoopRound;
    return addrOffset;
}

uint16_t CcuSimulator::GetLoopMsOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.msOffset_ * loopGroupInfo_.loopStatus_.loopExtendIndex;
}

uint16_t CcuSimulator::GetLoopCKEOffset()
{
    if (state_ != CcuExecState::EXEC_LOOP_INSTR) {
        return 0;
    }
    return loopGroupInfo_.ckeOffset_ * loopGroupInfo_.loopStatus_.loopExtendIndex;
}

void CcuSimulator::SetExecState(CcuExecState state)
{
    state_ = state;
}

bool CcuSimulator::UpdateLoopStatus()
{
    if (waitCKE_) {
        // 如果是等待CKE的状态，那么就不更新状态，下次执行的时候还是等待CKE
        return false;
    }
    if (state_ == CcuExecState::EXEC_LOOPGROUP_INSTR) {
        return true;
    }
    if (state_ == CcuExecState::EXEC_NORMAL_INSTR || state_ == CcuExecState::EXEC_LOOP_INSTR) {
        curInstrId_++;
        if (curInstrId_ == endInstrId_) {
            state_ = CcuExecState::EXEC_SUCCESS;
            return true;
        }
    } else if (state_ == CcuExecState::EXEC_JUMP_INSTR) {
        curInstrId_ = jumpInstrId_;
        state_ = CcuExecState::EXEC_NORMAL_INSTR;
    }
    return true;
}

void CcuSimulator::InitJumpStatus(uint16_t jumpInstrId)
{
    state_ = CcuExecState::EXEC_JUMP_INSTR;
    jumpInstrId_ = jumpInstrId;
}

// 参数超过13个时，分为多个SQE任务下发：新SQE任务执行前，须初始化模拟器参数。
void CcuSimulator::Init(uint16_t startInstrId, uint16_t endInstrId, uint16_t instrCnt)
{
    if (finshed_ == false) {
        return;
    }
    finshed_ = false;
    startInstrId_ = startInstrId;
    endInstrId_   = endInstrId;
    instrCnt_     = instrCnt;
    curInstrId_   = startInstrId;
    state_ = CcuExecState::EXEC_NORMAL_INSTR;
}

CcuExecState CcuSimulator::GetState()
{
    return state_;
}

void CcuSimulator::SetWaitCKEFlag(bool needCKE)
{
    waitCKE_ = needCKE;
}

bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(dieId_);
    HCCL_DEBUG("[CcuSimulator][ExecuteInstr] locCcu[%d:%d], start instr[%u], state=[%d]", rankId_, dieId_, curInstrId, static_cast<int>(state_));
    if (curInstrId >= endInstrId_) {
        state_ =  CcuExecState::EXEC_FAIL;
        HCCL_ERROR("[CcuSimulator][ExecuteInstr] locCcu[%d:%d], Invalid curInstrId_[%u], endInstrId[%d]", rankId_, dieId_, curInstrId, endInstrId_);
        return false;
    }
    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(instrData[curInstrId].header.header, 0, rankId_, dieId_, instrData[curInstrId], this);
    if (executor == nullptr) {
        UpdateLoopStatus();
        HCCL_ERROR("[CcuSimulator][ExecuteInstr] locCcu[%d:%d], curInstrId_[%u] executor is null.", rankId_, dieId_, curInstrId);
        return true;  // 让还未添加的Executor先跑下去，看下是否能从start->loop->end
    }
    executor->Parser();
    executor->Run();
    if (UpdateLoopStatus() == false) {
        return false;
    }
    HCCL_DEBUG("[CcuSimulator][ExecuteInstr] locCcu[%d:%d], end instr[%d], %s", rankId_, dieId_, curInstrId, executor->Describe().c_str());
    return true;
}

// loop循环内指令执行，后续改为多线程并行执行
bool CcuSimulator::ExecuteLoop()
{
    auto instrCnt = loopGroupInfo_.loopStatus_.loopEndInstrId - loopGroupInfo_.loopStatus_.loopStartInstrId + 1;
    for (uint32_t i = 0; i < loopGroupInfo_.loopStatus_.loopExecCount; i++) {
        HCCL_DEBUG("[CcuSimulator][ExecuteLoop] locCcu[%d:%d], loopStartInstrId=[%u], loopCurInstrId=[%u], instrCnt=[%u], loopRound=[%u].",
            rankId_, dieId_, loopGroupInfo_.loopStatus_.loopStartInstrId, curInstrId_, instrCnt, i);
        auto simulator = std::make_unique<CcuSimulator>(
            rankId_, dieId_, loopGroupInfo_.loopStatus_.loopStartInstrId, loopGroupInfo_.loopStatus_.loopEndInstrId + 1, instrCnt);
        loopGroupInfo_.loopStatus_.curLoopRound = i; // 设置当前迭代次数
        simulator->InitLoopGroupInfo(loopGroupInfo_);
        simulator->SetExecState(CcuExecState::EXEC_LOOP_INSTR);
        simulator->Execute();
    }
    return true;
}

// loopGroup内所有展开的loop执行，后续改为多线程并行执行
bool CcuSimulator::ExecuteLoopGroup()
{
    HCCL_DEBUG("[CcuSimulator][ExecuteLoopGroup] locCcu[%d:%d], lopNum=[%u], loopExtendNum=[%u], loopOffset=[%u].",
        rankId_, dieId_, loopGroupInfo_.loopNum_, loopGroupInfo_.loopExtendNum_, loopGroupInfo_.loopOffset_);
    if (loopGroupInfo_.loopNum_ == 0) {
        return true;
    }
    if (loopGroupInfo_.loopNum_ > 2) { // 2: 目前最多支持两个loop模板
        return false;
    }

    // 原始loop模板 + 展开loop的执行
    for (uint32_t idx = 0; idx <= loopGroupInfo_.loopExtendNum_; idx++) {
        // idx = 0时，原始loop模板：从0开始执行loop，展开index = 0；
        // idx ≠ 0时，展开loop指令：从loopOffset开始执行，展开index = idx。
        int loopStartIdx = (idx == 0 ? 0 : loopGroupInfo_.loopOffset_);
        for (uint32_t i = loopStartIdx; i < loopGroupInfo_.loopNum_; i++) {
            loopGroupInfo_.loopStatus_.loopExtendIndex = idx;
            ExecuteInstr(loopGroupInfo_.startLoopId_ + i);
        }
    }
    return true;
}

bool CcuSimulator::Execute()
{
    while (curInstrId_ != endInstrId_) {
        if (ExecuteInstr(curInstrId_)) {
            continue;
        }
        return false;
    }
    if (curInstrId_ == endInstrId_) {
        finshed_ = true;
    }
    // CcuResouceManager::GetInstance().DumpCcuXnResouceInfo();
    return true;
}