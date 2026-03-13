/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load sqe arg to gsa
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoadSqeArgsToGsaExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册LoadSqeArgsToGsaExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADSQEARGSTOGSA_CODE, LoadSqeArgsToGsaExecutor)

void LoadSqeArgsToGsaExecutor::Parser() {
    gsaId_    = instr_.v1.loadSqeArgsToXn.xnId;
    sqeArgId_ = instr_.v1.loadSqeArgsToXn.sqeArgsId;
}

void LoadSqeArgsToGsaExecutor::Run() {
    // 加载sqe参数至xn寄存器，只需更新对应dev的ccu资源映射表即可
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    uint64_t sqeArgValue = ccuResMgr.GetSqeArgValue(rankId_, dieId_, sqeArgId_);

    ccuResMgr.UpdateXnValue(rankId_, dieId_, gsaId_, sqeArgValue);
}

std::string LoadSqeArgsToGsaExecutor::Describe() {
    return Hccl::StringFormat("[Simulation Execute] Load SqeArg[%u] to GSA[%u]", sqeArgId_, gsaId_);
}