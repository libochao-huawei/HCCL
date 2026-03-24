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
using namespace ops_hccl;
HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode **opParam);
HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode *opParam);
HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode *opParam, const char *opType);
HcclResult HcclSetOpParamGraphModeDataCount(OpParamGraphMode *opParam, const u64 *dataCount);
HcclResult HcclSetOpParamGraphModeDataType(OpParamGraphMode *opParam, HcclDataType dataType);
HcclResult HcclSetOpParamGraphModeRankSize(OpParamGraphMode *opParam, const u32 *rankSize);
HcclResult HcclSetOpParamGraphModeHCCLBufferSize(OpParamGraphMode *opParam, const u64 *hcclBufferSize);
HcclResult HcclCalcOpResOnlineGraphMode(OpParamGraphMode *opParam, u64 *opMemSize, u32 *streamNum, u32 *taskNum, u32 *aivCoreNum);
HcclResult HcclCalcOpResOfflineGraphMode(OpParamGraphMode *opParam, u64 *opMemSize, u32 *streamNum, u32 *taskNum, u32 *aivCoreNum);

#ifdef __cplusplus
}
#endif
namespace ops_hccl {
HcclResult CheckCalcResInputGraphMode(const OpParamGraphMode *opParam, const u64 *opMemSize, const u32 *streamNum, const u32 *taskNum, const u32 *aivCoreNum);
HcclResult HcclCalcAicpuResOffline(ResResponseGraphMode *resResponse);
HcclResult HcclCalcCcuResOffline(OpParamGraphMode *opParam, ResResponseGraphMode *resResponse);
HcclResult CalcTaskNum(OpParamGraphMode *opParam, u32 &ccuTaskNum);

} // namespace ops_hccl
