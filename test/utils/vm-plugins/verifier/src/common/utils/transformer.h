/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_TRANSFORMER_H
#define HCCL_SIM_TRANSFORMER_H

#include "checker_def.h"
#include "hccl/hccl_types.h"
#include "checker_def.h"
#include "sim_common.h"
#include <map>
#include <vector>

extern std::map<CheckerOpType, HcclCMDType> g_CheckerOpType2HcclCMDType;
extern std::map<CheckerReduceOp, HcclReduceOp> g_CheckerReduceOp2HcclReduceOp;
extern std::map<CheckerDataType, HcclDataType> g_CheckerDataType2HcclDataType;
extern std::map<CheckerDataType, uint32_t> g_CheckerDataTypeSize;

extern std::vector<u32> sizeTable;
extern std::map<HcclReduceOp, CheckerReduceOp> g_HcclReduceOp2CheckerReduceOp;
extern std::map<HcclDataType, CheckerDataType> g_HcclDataType2CheckerDataType;

#endif