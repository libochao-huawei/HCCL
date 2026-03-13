/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 参数转换头文件，用于对接可视化工具及命令行工具
 * Author: yinding
 * Create: 2025-01-15
 */

#ifndef SIMPLE_PARAM_H
#define SIMPLE_PARAM_H

#include "checker_def.h"

namespace checker {

struct SimpleParam {
    CheckerOpType opType { CheckerOpType::ALLREDUCE };
    std::string algName;
    CheckerOpMode opMode { CheckerOpMode::OPBASE };
    CheckerReduceOp reduceType { CheckerReduceOp::REDUCE_RESERVED };
    CheckerDevType devtype { CheckerDevType::DEV_TYPE_910B };
    bool is310P3V = false;  // 仅当310PV卡的时候，设置为1
    u32 root { 0 };
    u32 dstRank { 1 };
    u32 srcRank { 0 };
    u64 count { 160 };
    CheckerDataType dataType { CheckerDataType::DATA_TYPE_FP32 };
};

HcclResult GenTestOpParams(u32 rankSize, const SimpleParam& uiParams, CheckerOpParam& checkerParams);

}

#endif