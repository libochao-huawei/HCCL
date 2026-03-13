/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load imd to gsa
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoadImdToGSAExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册LoadImdToGSAExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADIMDTOGSA_CODE, LoadImdToGSAExecutor)

void LoadImdToGSAExecutor::Parser()
{
    gsaId_      = instr_.v1.loadImdToGSA.gsaId;
    immediate_  = instr_.v1.loadImdToGSA.immediate;
}

void LoadImdToGSAExecutor::Run()
{
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    ccuResMgr.UpdateGsaValue(rankId_, dieId_, gsaId_, immediate_);
}

std::string LoadImdToGSAExecutor::Describe()
{
    return Hccl::StringFormat("[Simulation Execute] Load immediate[%llu] to GSA[%u]", immediate_, gsaId_);
}