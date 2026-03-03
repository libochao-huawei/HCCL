/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 算法库AllGatherParallelExecutor类实现
 * Author: xxx
 * Create: 2026-xx-xx
 */

#ifndef HCCLV2_INS_ALL_GATHER_CONCURR_EXECUTOR_H
#define HCCLV2_INS_ALL_GATHER_CONCURR_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsAllGatherConcurrentExecutor : public InsCollAlgBase {
public:
    explicit InsAllGatherConcurrentExecutor();
    ~InsAllGatherConcurrentExecutor() = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfo *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

private:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                               InsAlgTemplate0 &algTemplateMesh, InsAlgTemplate1 &algTemplateNhr);

    HcclResult InitCommInfo(const OpParam &param, const TopoInfo *topoInfo,
                            const AlgHierarchyInfoForAllLevel &algHierarchyInfo);

    void GetParallelDataSplit(std::vector<float> &splitDataSize) const;

    void GenAlgParamsforTemplate0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 dataCountPerLoopMesh, const u64 scratchOffset,
                                  TemplateDataParams &tempAlgParamsMesh) const;

    void GenAlgParamsforTemplate1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                 const u64 dataCountPerLoopNhr, const u64 scratchOffset,
                                 TemplateDataParams &tempAlgParamsNhr) const;

    HcclResult PrepareResForTemplate(const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &algTemplateMesh, InsAlgTemplate1 &algTemplateNhr);

    std::vector<ThreadHandle> threads_;  // 相当于之前的std::vector<InsQuePtr> tempInsQue_;
    std::vector<ThreadHandle> tmp0Threads_;
    std::vector<ThreadHandle> tmp1Threads_;
    ThreadHandle mainThread_;
    std::vector<ThreadHandle> templateMainThreads_;
    std::vector<u32> syncNotifyOnTemplates_;
    std::vector<u32> syncNotifyOnMain_;

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::map<u32, std::vector<ChannelInfo>> tmp0LinkMap_;
    std::map<u32, std::vector<ChannelInfo>> tmp1LinkMap_;
    std::vector<CcuKernelHandle> tmp0CcuKernels_;
    std::vector<CcuKernelHandle> tmp1CcuKernels_;
};
}

#endif  // HCCLV2_INS_ALL_GATHER_CONCURR_EXECUTOR_H