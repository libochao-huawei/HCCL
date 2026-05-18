/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_HCCL_OP_COMMON_A3
#define OPS_HCCL_OP_COMMON_A3

#include "alg_param.h"
#include "executor_base.h"
#include "alg_type.h"
#include "op_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

namespace ops_hccl {

HcclResult ProcessA3(HcclComm comm, OpParam& param, uint64_t beginTime);

HcclResult ExecOp(HcclComm comm, OpParam &param);

HcclResult SelectAlgScatter(HcclComm comm, OpParam &param, TopoInfo* topoInfo, AlgType& algType, std::string &algName);

HcclResult SetAlgoLevel0(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel0 &algType);

HcclResult GetDefaultAlgoLevel0Module(TopoInfo* topoInfo, AlgTypeLevel0 &algType);

HcclResult SetAlgoLevel1(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel1 &algType,
	HcclCMDType opType);

HcclResult GetDefaultAlgoLevel1V1(TopoInfo* topoInfo, AlgTypeLevel1 &algType);

HcclResult SetAlgoLevel2(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel2 &algType);

bool IsStreamCapture(aclrtStream stream);

bool IsAiCpuMode(DevType deviceType, u32 rankSize);

HcclResult CalcBaseTopoInfo(HcclComm comm, OpParam &param, TopoInfo** topoInfo);

HcclResult GetAlgType(TopoInfo* topoInfo, HcclCMDType opType, AlgType& algType);

HcclResult AllocAlgResource(HcclComm comm, const OpParam& param, AlgResourceRequest &resRequest,
	AlgResourceCtx* resCtxHost);

std::string SetLaunchMode(CommEngine engine);

HcclResult ReportProfilingThread(HcclComm comm, const OpParam &param, AlgResourceCtx *resCtxHost, TopoInfo* topoInfo);

}

#endif