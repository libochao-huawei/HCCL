
#ifndef HCCL_INS_SEND_EXECUTOR_H
#define HCCL_INS_SEND_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {
    template <typename InsAlgTemplate>
    class InsSendDpuExecutor : public InsCollAlgBase {
    public:
        std::string Describe() const override;
        // 算法编排
        HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

        HcclResult CalcAlgHierarchyInfo(
            HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;
        
        // 资源计算
        HcclResult CalcRes(
            HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
            const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;
    
    protected:
        HcclResult InitCommInfo(
            HcclComm comm, const OpParam &param, TopoInfoWithNetLayerDetails *topoInfo,
            const AlgHierarchyInfoForAllLevel &algHierarchyInfo);
        // 单算子还是图模式
        OpMode opMode_;
        u32 remoteRank;
        u32 myRank_;
        std::vector<ThreadHandle> threads_;
        // 一次搬运最大数据量
        u64 maxLoopTransSize_;
        // 一次搬运最大数据个数
        u64 maxLoopTransCount_;

        AlgHierarchyInfoForAllLevel algHierarchyInfo_;
        std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    };
} // namespace ops_hccl

#endif //HCCL_INS_SEND_EXECUTOR_H
