/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor create func mgr
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "CcuExecutorManager.h"

using namespace std;

CcuExecutorCreateFuncMgr& CcuExecutorCreateFuncMgr::Instance() {
    static CcuExecutorCreateFuncMgr instance;
    return instance;
}