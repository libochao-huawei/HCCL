/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- reduce add
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include <map>
#include "string_util.h"
#include "ReduceAddExecutor.h"
#include "CcuExecutorManager.h"
#include "SimulatorBase.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册ReduceAddExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::REDUCE_TYPE, SimCcuV1::ADD_CODE, ReduceAddExecutor)

void ReduceAddExecutor::Parser()
{
    count_       = instr_.v1.add.count;
    castEn_      = instr_.v1.add.castEn;
    dataType_    = instr_.v1.add.dataType;
    clearType_   = instr_.v1.add.clearType;
    setCKEId_    = instr_.v1.add.setCKEId;
    setCKEMask_  = instr_.v1.add.setCKEMask;
    waitCKEId_   = instr_.v1.add.waitCKEId;
    waitCKEMask_ = instr_.v1.add.waitCKEMask;
    (void)memcpy_s(msId_, sizeof(uint16_t) * CCU_REDUCE_MAX_MS, instr_.v1.add.msId, sizeof(uint16_t) * CCU_REDUCE_MAX_MS);
}

// FP32/INT16/INT32/UINT8类型
template <typename T>
void ReduceAdd(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < BYTE_NUM_4K / sizeof(T); j++) {
            *(msValuePtr0 + j) += *(msValuePtr + j);
        }
    }
}

template <typename T>
void ReduceAddFp16(int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn)
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < BYTE_NUM_4K / sizeof(T); j++) {
            *(msValuePtr0 + j) += *(msValuePtr + j);
        }
    }
}

const std::map<ReduceAddDataType, std::function<void(int, int, uint16_t[], uint16_t, uint16_t)>> reduceAddFuncMap =
{
    {ReduceAddDataType::ADD_INT16,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) { ReduceAdd<s16>(rankId, dieId, msId, count); }},
    {ReduceAddDataType::ADD_INT32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) { ReduceAdd<s32>(rankId, dieId, msId, count); }},
    {ReduceAddDataType::ADD_UINT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) { ReduceAdd<uint8_t>(rankId, dieId, msId, count); }},
    {ReduceAddDataType::ADD_INT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) { ReduceAdd<s8>(rankId, dieId, msId, count); }},
    {ReduceAddDataType::ADD_FP32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) { ReduceAdd<float>(rankId, dieId, msId, count); }},
    {ReduceAddDataType::ADD_FP16,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count, uint16_t castEn) { ReduceAddFp16<FP16>(rankId, dieId, msId, count, castEn); }}
};

// Reduce Add操作
void ReduceAddExecutor::Process(CcuResouceManager &ccuResMgr)
{
    HCCL_DEBUG("[ReduceAddExecutor][Process] Reduce Add info, locCcu[%d:%d], count=[%u], castEn=[%d], dataType=[%d]",
        rankId_, dieId_, count_, castEn_, dataType_);
    for (uint32_t i = 0; i < CCU_REDUCE_MAX_MS; i++) {
        HCCL_DEBUG("msId_[%u]:dieId[%u], msId[%u]", i, msId_[i] >> 15, msId_[i] & 0x7FFF);
        msId_[i] = msId_[i] & 0x7FFF;
    }
    if (dataType_ >= ReduceAddDataType::ADD_RESERVED) {
        return;
    }
    // 1.判断是否在Loop循环内
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        for (uint32_t i = 0; i < Hccl::CcuRep::CCU_REDUCE_MAX_MS; i++) {
            msId_[i] += msOffset;
        }
        HCCL_DEBUG("[ReduceAddExecutor][Process][INFO] locCcu[%d:%d], ms offset=[%u]", rankId_, dieId_, msOffset);
    }
    // 2. reduce操作
    ReduceAddDataType type = static_cast<ReduceAddDataType>(dataType_);
    auto res = reduceAddFuncMap.find(type);
    if (res !=  reduceAddFuncMap.end()) {
        res->second(rankId_, dieId_, msId_, count_, castEn_);
    }
    // 3.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void ReduceAddExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "ReduceAdd");
}

std::string ReduceAddExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Add %s with Count[%u], DataType[%u] and "
                              "CastEn[%u], Set CKE[%u:%04x], clearType[%u]",
        waitCKEId_,
        waitCKEMask_,
        ParseMSList().c_str(),
        count_,
        dataType_,
        castEn_,
        setCKEId_,
        setCKEMask_,
        clearType_);
}