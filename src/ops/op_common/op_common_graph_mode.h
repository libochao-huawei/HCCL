/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OP_COMMON
#define OPS_HCCL_OP_COMMON

#include <string>
#include <memory>
#include "hccl.h"
#include "op_common.h"
#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

namespace ops_hccl {

HcclResult HcclExecOpGraphMode(HcclComm comm, OpParam &param, ResPackGraphMode &resPack);

HcclResult HcclRegstryBuffGraphMode(HcclComm comm, const char *memTag, void *bufferPtr, u64 buffSize);

HcclResult HcclGetRemoteBuffGraphMode(HcclComm comm, ChannelHandle channel, const char *memTag, void **bufferPtr, u64 *buffSize);

HcclResult HcclGetAlgResGraphMode(HcclComm comm, OpParam &param, std::shared_ptr<InsCollAlgBase> &executor, TopoInfo *topoInfo,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void **resCtxSequence, bool &isResourceReused);

HcclResult GetAlgResAICPUGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag);

HcclResult HcclAllocAlgResourceAICPUGraphMode(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

HcclResult HcclGetChannelGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

HcclResult HcclGetCcuKernelGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclGetChannelForCcuGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclAllocAlgResourceCcuGraphMode(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                                   std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);
HcclResult GetAlgResCcuGraphMode(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                        std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo* topoInfo,
                        AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize);

HcclResult GetAlgResAivGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize);

HcclResult HcclAllocAlgResourceAivGraphMode(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, AlgResourceCtxSerializable* resCtxHost);

HcclResult GetAlgResDPUGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag);
}  // namespace ops_hccl

#endif