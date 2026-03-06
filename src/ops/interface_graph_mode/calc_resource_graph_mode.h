/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_param.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode** opParam);
HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode opParam);
HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode opParam, const char* opType);
HcclResult HcclCalcOpResOnlineGraphMode(OpParamGraphMode* opParam, u64* opMemSize, u32* streamNum, u32* taskNum, u32* aivCoreNum);
HcclResult HcclCalcOpResOfflineGraphMode(OpParamGraphMode* opParam, u64* opMemSize, u32* streamNum, u32* taskNum, u32* aivCoreNum);

#ifdef __cplusplus
}
#endif
namespace hccl {
HcclResult HcclCalcAicpuResOffline(ResResponseGraphMode *resResponse);
} // namespace hccl
