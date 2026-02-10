/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load gsa to xn
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoadGsaXnExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册LoadGSAXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADGSAXN_CODE, LoadGsaXnExecutor)

void LoadGsaXnExecutor::Parser()
{
    gsAdId_ = instr_.v1.loadGSAXn.gsAdId;
    gsAmId_ = instr_.v1.loadGSAXn.gsAmId;
    xnId_ = instr_.v1.loadGSAXn.xnId;
}

void LoadGsaXnExecutor::Run()
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    uint64_t gsaVal = ccuResMgr.GetGsaValue(rankId_, dieId_, gsAmId_);
    uint64_t xnVal = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
    uint64_t val = gsaVal + xnVal;
    ccuResMgr.UpdateGsaValue(rankId_, dieId_, gsAdId_, val);
}

std::string LoadGsaXnExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Load GSA[%u] + Xn[%u] to GSA[%u]", gsAmId_, xnId_, gsAdId_);
}