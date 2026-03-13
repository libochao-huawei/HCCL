/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- load sqe arg to xn
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "string_util.h"
#include "LoadSqeArgsToXnExecutor.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"

using namespace std;
using namespace Hccl::CcuRep;

// 注册LoadSqeArgsToXnExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::LOAD_TYPE, SimCcuV1::LOADSQEARGSTOXN_CODE, LoadSqeArgsToXnExecutor)

void LoadSqeArgsToXnExecutor::Parser() {
    xnId_     = instr_.v1.loadSqeArgsToXn.xnId;
    sqeArgId_ = instr_.v1.loadSqeArgsToXn.sqeArgsId;
}

void LoadSqeArgsToXnExecutor::Run() {
    // 加载sqe参数至xn寄存器，只需更新对应dev的ccu资源映射表即可
    auto &ccuResMgr = CcuResouceManager::GetInstance();
    uint64_t sqeArgValue = ccuResMgr.GetSqeArgValue(rankId_, dieId_, sqeArgId_);

    HCCL_DEBUG("[LoadSqeArgsToXnExecutor] Load arg: locCcu[%d:%d], XnId=[%u], argId=[%u], value=[%u]",
        rankId_, dieId_, xnId_, sqeArgId_, sqeArgValue);
    ccuResMgr.UpdateXnValue(rankId_, dieId_, xnId_, sqeArgValue);
}

std::string LoadSqeArgsToXnExecutor::Describe() {
    uint64_t sqeArgValue = CcuResouceManager::GetInstance().GetSqeArgValue(rankId_, dieId_, sqeArgId_);
    return Hccl::StringFormat("[Simulation Execute] locCcu[%d:%d], Load SqeArg[%u]-Value[%x] to Xn[%u]", rankId_, dieId_, sqeArgId_, sqeArgValue, xnId_);
}