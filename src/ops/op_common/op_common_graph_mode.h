/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OP_COMMON_GRAPH_MODE
#define OPS_HCCL_OP_COMMON_GRAPH_MODE

#include <string>
#include <memory>
#include "hccl.h"
#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"
#include "op_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

namespace ops_hccl {

HcclResult HcclExecOpGraphMode(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, std::string &algName, const ResPackGraphMode &resPack);

HcclResult HcclRegstryBuffGraphMode(HcclComm comm, const char *memTag, void *bufferPtr, uint64_t bufferSize, HcclMemHandle *memHandle);

HcclResult HcclGetRemoteBuffGraphMode(HcclComm comm, ChannelHandle channel, const char *memTag, void **bufferPtr, uint64_t *bufferSize);

HcclResult HcclGetAlgResGraphMode(HcclComm comm, OpParam &param, std::shared_ptr<InsCollAlgBase> &executor, TopoInfoWithNetLayerDetails *topoInfo,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void **resCtxSequence, bool &isResourceReused);

HcclResult GetAlgResAICPUGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag);

HcclResult HcclAllocAlgResourceAICPUGraphMode(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

HcclResult HcclGetChannelGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

}  // namespace ops_hccl

#endif // OPS_HCCL_OP_COMMON_GRAPH_MODE