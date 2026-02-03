/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_SCATTER_OP
#define OPS_HCCL_SRC_OPS_SCATTER_OP

#include <string>
#include "hccl.h"

#include "alg_param.h"
#include "executor_base.h"
#include "alg_type.h"

#ifdef __cplusplus
extern "C" {
#endif
constexpr u32 MAX_LENGTH = 128;

typedef struct HcomProInfo {
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
}HcomProInfo;

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult ScatterOutPlace(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream, const std::string &tag);

HcclResult ExecOp(HcclComm comm, OpParam &param);

HcclResult CalcBaseTopoInfo(HcclComm comm, OpParam &param, TopoInfo** topoInfo);

HcclResult SelectAlg(HcclComm comm, OpParam &param, TopoInfo* topoInfo, AlgType& algType, std::string &algName);

HcclResult GetAlgRes(HcclComm comm, OpParam &param, std::unique_ptr<ExecutorBase> &executor,
    TopoInfo* topoInfo, AlgType& algType, AlgResourceCtx** resCtx);

HcclResult GetAlgType(TopoInfo* topoInfo, HcclCMDType opType, AlgType& algType);

HcclResult AllocAlgResource(HcclComm comm, const OpParam& param, AlgResourceRequest &resRequest,
    AlgResourceCtx* resCtxHost);

HcclResult SetAlgoLevel0(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel0 &algType);

HcclResult GetDefaultAlgoLevel0Module(TopoInfo* topoInfo, AlgTypeLevel0 &algType);

HcclResult SetAlgoLevel1(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel1 &algType,
    HcclCMDType opType);

HcclResult GetDefaultAlgoLevel1V1(TopoInfo* topoInfo, AlgTypeLevel1 &algType);

HcclResult SetAlgoLevel2(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel2 &algType);

bool IsStreamCapture(aclrtStream stream);

bool IsAiCpuMode(DevType deviceType, u32 rankSize);

HcclResult CheckCount(const u64 count);

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);

HcclResult CheckScatterInputPara(HcclComm comm, void *recvBuf);

std::string SetLaunchMode(CommEngine engine);

HcclResult ReportProfilingThread(const OpParam &param, AlgResourceCtx *resCtxHost, TopoInfo* topoInfo);

}

#endif