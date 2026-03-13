/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 适配1.0的类型转换接口
 * Author: huangweihao, yinding
 * Create: 2025-05-06
 */

#ifndef HCCL_ADAPTER_V1_TRANSFORMER_H
#define HCCL_ADAPTER_V1_TRANSFORMER_H

#include <map>
#include <unordered_map>
#include <vector>

#include "checker_def.h"
#include "hccl_common.h"
#include "alg_cmd_type.h"

using namespace checker;

namespace hccl {

extern std::map<CheckerOpType, HcclCMDType> g_CheckerOpType2HcclCMDType;
extern std::map<HcclCMDType, CheckerOpType> g_HcclCMDType2CheckerOpType;
extern std::map<CheckerReduceOp, HcclReduceOp> g_CheckerReduceOp2HcclReduceOp;
extern std::map<HcclReduceOp, CheckerReduceOp> g_HcclReduceOp2CheckerReduceOp;
extern std::map<CheckerDataType, HcclDataType> g_CheckerDataType2HcclDataType;
extern std::map<HcclDataType, CheckerDataType> g_HcclDataType2CheckerDataType;
extern std::map<CheckerDevType, DevType> g_CheckerDevType2HcclDevType;
extern std::map<DevType, CheckerDevType> g_HcclDevType2CheckerDevType;

} // namespace checker

#endif