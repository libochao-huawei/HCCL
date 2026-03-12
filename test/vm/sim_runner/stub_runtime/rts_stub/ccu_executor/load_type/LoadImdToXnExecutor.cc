/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load imd to xn
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoadImdToXnExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册LoadImdToXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADIMDTOXN_CODE, LoadImdToXnExecutor)

void LoadImdToXnExecutor::Parser()
{
    xnId_       = instr_.v1.loadImdToXn.xnId;
    immediate_  = instr_.v1.loadImdToXn.immediate;
}

void LoadImdToXnExecutor::Run()
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xnId_, immediate_);
}

std::string LoadImdToXnExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] locCcu[%d:%d], Load immediate[%llu] to Xn[%u]", rankId_, dieId_, immediate_, xnId_);
}