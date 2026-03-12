/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load gsa to gsa
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoadGsaGsaExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册LoadGSAGSAExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADGSAGSA_CODE, LoadGsaGsaExecutor)

void LoadGsaGsaExecutor::Parser()
{
    gsAdId_ = instr_.v1.loadGSAGSA.gsAdId;
    gsAmId_ = instr_.v1.loadGSAGSA.gsAmId;
    gsAnId_ = instr_.v1.loadGSAGSA.gsAnId;
}

void LoadGsaGsaExecutor::Run()
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    uint64_t gsa1 = ccuResMgr.GetGsaValue(rankId_, dieId_, gsAmId_);
    uint64_t gsa2 = ccuResMgr.GetGsaValue(rankId_, dieId_, gsAnId_);
    uint64_t val = gsa1 + gsa2;
    ccuResMgr.UpdateGsaValue(rankId_, dieId_, gsAdId_, val);
}

std::string LoadGsaGsaExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Load GSA[%u] + GSA[%u] to GSA[%u]", gsAmId_, gsAnId_, gsAdId_);
}