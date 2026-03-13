/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- loadXX
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoadXnXnExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册LoadGSAGSAExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADXX_CODE, LoadXnXnExecutor)

void LoadXnXnExecutor::Parser()
{
    xdId_ = instr_.v1.loadXX.xdId;
    xmId_ = instr_.v1.loadXX.xmId;
    xnId_ = instr_.v1.loadXX.xnId;
}

void LoadXnXnExecutor::Run()
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    uint64_t xn1 = ccuResMgr.GetXnValue(rankId_, dieId_, xmId_);
    uint64_t xn2 = ccuResMgr.GetXnValue(rankId_, dieId_, xnId_);
    uint64_t val = xn1 + xn2;
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xdId_, val);
}

std::string LoadXnXnExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Load Xn[%u] + Xn[%u] to Xn[%u]", xmId_, xnId_, xdId_);
}