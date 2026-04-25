/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INS_V2_ALL_GATHER_CCU_OMNIPIPE_EXECUTOR_H
#define HCCL_INS_V2_ALL_GATHER_CCU_OMNIPIPE_EXECUTOR_H

#include "executor_common_ops.h"
#include "omnipipe_data_slice_calc.h"
#include "common_alg_template_base.h"

namespace ops_hccl {

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
class InsV2AllGatherCcuOmniPipeExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllGatherCcuOmniPipeExecutor();
    ~InsV2AllGatherCcuOmniPipeExecutor() override = default;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
    HcclResult FastLaunch(const OpParam &param, const CcuFastLaunchCtx *ctx) override;

private:
    HcclResult InitCommInfo(const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                            const AlgHierarchyInfoForAllLevel &algHierarchyInfo);
    HcclResult InitExecutorInfo();
    HcclResult CalcResLevel(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                            const std::shared_ptr<CommonAlgTemplateBase> &tempAlg,
                            AlgResourceRequest &resourceRequest);
    HcclResult PrepareResForTemplateLevel(u32 level, const std::shared_ptr<CommonAlgTemplateBase> &tempAlg,
                                          u32 &ccuKernelNumIdx, u32 &ccuKernelHandleOffset,
                                          const AlgResourceCtxSerializable &resCtx,
                                          TemplateResource &templateResource);
    HcclResult GenTemplateAlgParamsByDimData(TemplateDataParams &tempAlgParams,
                                              const StepSliceInfo &stepSliceInfo) const;
    HcclResult BuildEndpointAttrBw(const OpParam &param,
                                   std::vector<EndpointAttrBwCoeff> &endpointAttrBwNew) const;
    HcclResult CalcMaxCountPerLoop(u64 &maxCountPerLoop) const;
    OmniPipeSliceInfo CalcLoopSliceInfo(u64 currDataCount,
                                        const std::vector<EndpointAttrBwCoeff> &endpointAttrBw) const;
    HcclResult RunOrchestrateLevel2(const OpParam &param, std::shared_ptr<CommonAlgTemplateBase> tempAlg,
                                    TemplateDataParams &params, TemplateResource &resource,
                                    TemplateResource &orderedFastLaunchInfos);
    HcclResult RunOrchestrateLevelXY(const OpParam &param, const OmniPipeSliceInfo &sliceInfo, u32 idx,
                                     std::map<u32, std::shared_ptr<CommonAlgTemplateBase>> &tempMap,
                                     std::map<u32, TemplateResource> &tempResMap,
                                     TemplateDataParams &paramsLevel0, TemplateDataParams &paramsLevel1,
                                     TemplateResource &orderedFastLaunchInfos);
    HcclResult RunLoopCopyOut(const OpParam &param, const HcclMem &hcclBuff, u64 currBytes,
                              u64 processedDataCount, TemplateResource &copyResource,
                              TemplateResource &orderedFastLaunchInfos) const;
    HcclResult RunOrchestrateOneLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                     const std::vector<EndpointAttrBwCoeff> &endpointAttrBw, u64 currDataCount,
                                     u64 processedDataCount,
                                     std::map<u32, std::shared_ptr<CommonAlgTemplateBase>> &tempMap,
                                     std::map<u32, TemplateResource> &tempResMap,
                                     TemplateResource &orderedFastLaunchInfos);
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                               std::map<u32, std::shared_ptr<CommonAlgTemplateBase>> &tempMap,
                               std::map<u32, TemplateResource> &tempResMap);
    HcclResult RunCcuLocalCopy(const OpParam &param, void *srcPtr, void *dstPtr, const HcclMem &hcclBuff,
                               u64 srcOffset, u64 dstOffset, u64 copySize, TemplateResource &copyResource,
                               TemplateResource *fastLaunchResource = nullptr) const;
    HcclResult FastLaunchSaveCtx(const OpParam &param, const TemplateResource &orderedSubmitInfos);
    HcclResult InitFastLaunchActionThreads(const CcuFastLaunchCtx *ctx,
                                           std::map<u32, std::vector<ThreadHandle>> &actionThreads);
    void InitFastLaunchNotifyInfo(const std::map<u32, std::vector<ThreadHandle>> &actionThreads);
    HcclResult SetFastLaunchCtxAddr(const OpParam &param, u32 action, const HcclMem &hcclBuff,
                                    TemplateFastLaunchCtx &tempCtx) const;
    HcclResult FastLaunchSubmitInfo(const OpParam &param, const CcuKernelSubmitInfo &submitInfo,
                                    const std::map<u32, std::vector<ThreadHandle>> &actionThreads,
                                    const HcclMem &hcclBuff, TemplateFastLaunchCtx &tempCtx);

    std::vector<uint64_t> rankSizeLevel_;
    std::vector<uint64_t> rankIdxLevel_;
    OpMode opMode_{OpMode::OPBASE};
    ThreadHandle controlThread_{0};
    std::vector<ThreadHandle> threads_;
    std::vector<std::vector<ThreadHandle>> levelThreads_;
    std::vector<ThreadHandle> tempMainThreadsXY_;
    std::vector<ThreadHandle> tempMainThreadsZ_;
    std::vector<u32> ntfIdxCtrlToTempXY_;
    std::vector<u32> ntfIdxTempToCtrlXY_;
    std::vector<u32> ntfIdxCtrlToTempZ_;
    std::vector<u32> ntfIdxTempToCtrlZ_;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
};
#endif

} // namespace ops_hccl

#endif // HCCL_INS_V2_ALL_GATHER_CCU_OMNIPIPE_EXECUTOR_H
