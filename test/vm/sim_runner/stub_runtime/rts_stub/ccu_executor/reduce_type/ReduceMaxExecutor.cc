/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- reduce max
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "ReduceMaxExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册ReduceMaxExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::REDUCE_TYPE, SimCcuV1::MAX_CODE, ReduceMaxExecutor)

void ReduceMaxExecutor::Parser()
{
    count_       = instr_.v1.max.count;
    dataType_    = instr_.v1.max.dataType;
    clearType_   = instr_.v1.max.clearType;
    setCKEId_    = instr_.v1.max.setCKEId;
    setCKEMask_  = instr_.v1.max.setCKEMask;
    waitCKEId_   = instr_.v1.max.waitCKEId;
    waitCKEMask_ = instr_.v1.max.waitCKEMask;
    (void)memcpy_s(msId_, sizeof(uint16_t) * CCU_REDUCE_MAX_MS, instr_.v1.max.msId, sizeof(uint16_t) * CCU_REDUCE_MAX_MS);
}

// INT16/INT32/UINT8/INT8类型
template <typename T>
void ReduceMax(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (ms0Value > msxValue)  ? ms0Value : msxValue;
        }
    }
}

// FP32类型
template <typename T>
void ReduceMaxFp32(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    float epsilon = 1e-5;
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (fabs(ms0Value - msxValue) >= epsilon)  ? ms0Value : msxValue;
        }
    }
}

// FP16类型
template <typename T>
void ReduceMaxFp16(int rankId, int dieId, uint16_t msId[], uint16_t count)
{
    float epsilon = 1e-5;
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    auto msValuePtr0 = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[0]));
    for (uint16_t i = 1; i < count + 2; i++) {
        auto msValuePtr = reinterpret_cast<T*>(ccuResMgr.GetMsAddr(rankId, dieId, msId[i]));
        for (uint16_t j = 0; j < BYTE_NUM_4K / sizeof(T); j++) {
            T ms0Value = *(msValuePtr0 + j);
            T msxValue = *(msValuePtr + j);
            *(msValuePtr0 + j) = (fabs(ms0Value.to_float() - msxValue.to_float()) >= epsilon)  ? ms0Value : msxValue;
        }
    }
}

const std::map<ReduceMaxMinDataType, std::function<void(int, int, uint16_t[], uint16_t)>> reduceAddFuncMap =
{
    {ReduceMaxMinDataType::MAX_MIN_INT16,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) { ReduceMax<int16_t>(rankId, dieId, msId, count); }},
    {ReduceMaxMinDataType::MAX_MIN_INT32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) { ReduceMax<int32_t>(rankId, dieId, msId, count); }},
    {ReduceMaxMinDataType::MAX_MIN_UINT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) { ReduceMax<uint8_t>(rankId, dieId, msId, count); }},
    {ReduceMaxMinDataType::MAX_MIN_FP32,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) { ReduceMaxFp32<float>(rankId, dieId, msId, count); }},
    {ReduceMaxMinDataType::MAX_MIN_INT8,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) { ReduceMax<int8_t>(rankId, dieId, msId, count); }},
    {ReduceMaxMinDataType::MAX_MIN_FP16,
        [](int rankId, int dieId, uint16_t msId[], uint16_t count) { ReduceMaxFp16<FP16>(rankId, dieId, msId, count); }}
};

// Reduce Max操作
void ReduceMaxExecutor::Process(CcuResouceManager &ccuResMgr)
{
    HCCL_DEBUG("[ReduceMaxExecutor][Process] Reduce Max info, locCcu[%d:%d], count:[%u], dataType:[%u]", rankId_, dieId_, count_, dataType_);
    if (dataType_ >= ReduceMaxMinDataType::MAX_MIN_RESERVED4 || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED1 ||
        dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED2 || dataType_ == ReduceMaxMinDataType::MAX_MIN_RESERVED3) {
        return;
    }
    for (uint32_t i = 0; i < CCU_REDUCE_MAX_MS; i++) {
        HCCL_DEBUG("msId_[%u]:dieId[%u], msId[%u]", i, msId_[i] >> 15, msId_[i] & 0x7FFF);
        msId_[i] = msId_[i] & 0x7FFF;
    }
    // 1. 判断是否在Loop循环内
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto msOffset   = ccuSimulator_->GetLoopMsOffset();
        for (uint32_t i = 0; i < Hccl::CcuRep::CCU_REDUCE_MAX_MS; i++) {
            msId_[i] += msOffset;
        }
        HCCL_DEBUG("[ReduceMaxExecutor][Process] locCcu[%d:%d], ms offset=[%u]", rankId_, dieId_, msOffset);
    }
    // 2. reduce操作
    ReduceMaxMinDataType type = static_cast<ReduceMaxMinDataType>(dataType_);
    auto res = reduceAddFuncMap.find(type);
    if (res !=  reduceAddFuncMap.end()) {
        res->second(rankId_, dieId_, msId_, count_);
    }
    // 3. 设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void ReduceMaxExecutor::Run()
{
    WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "ReduceMax");
}

std::string ReduceMaxExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Wait CKE[%u:%04x], Max %s with Count[%u], DataType[%u] and "
                              "CastEn[%u], Set CKE[%u:%04x], clearType[%u]",
        waitCKEId_,
        waitCKEMask_,
        ParseMSList().c_str(),
        count_,
        dataType_,
        setCKEId_,
        setCKEMask_,
        clearType_);
}