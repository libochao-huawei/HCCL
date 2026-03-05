/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_graph_mode_interface.h"

HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode** opParam);
HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode opParam);
HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode opParam, const char* opType);
HcclResult HcclCalcOpResOnlineGraphMode(OpParamGraphMode opParam, uint64_t* opMemSize, uint64_t* streamNum, uint64_t* taskNum, uint64_t* aivCoreNum);
HcclResult HcclCalcOpResOfflineGraphMode(OpParamGraphMode opParam, uint64_t* opMemSize, uint64_t* streamNum, uint64_t* taskNum, uint64_t* aivCoreNum);
namespace hccl {
HcclResult HcclCalcAicpuResOffline(ResResponseGraphMode *resResponse);
} // namespace hccl
