#ifndef HCCL_ALLGATHERBATCH_NHR_CORE_H
#define HCCL_ALLGATHERBATCH_NHR_CORE_H

#include <vector>

#include "common.h"

namespace ops_hccl_allgatherbatch {

struct NHRStepInfo {
    uint32_t step = 0;
    uint32_t fromRank = 0;
    uint32_t toRank = 0;
    uint32_t sliceCount = 0;
    std::vector<uint32_t> txItemOrder;
    std::vector<uint32_t> rxItemOrder;
};

class AllGatherNHRCore {
public:
    AllGatherNHRCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes);

    // NHR 子模板入口：校验当前窗口的通信资源，并把远端 rank 的窗口数据拉回本地 rank 槽位。
    HcclResult RunAsync();

private:
    HcclResult ValidateCommState() const;
    HcclResult ValidateChannelMetadata() const;
    HcclResult ValidateProtocolDistribution() const;
    uint32_t CalcStepNum(uint32_t rankSize) const;
    HcclResult GetStepInfo(uint32_t step, uint32_t nSteps, NHRStepInfo &stepInfo) const;
    HcclResult BuildStepPlan(std::vector<NHRStepInfo> &stepPlan) const;
    const ChannelResource *FindChannel(uint32_t remoteRank) const;
    uint8_t *GetRankBuffer(uint32_t rank) const;
    bool IsCrossServerChannel(const ChannelResource &channel) const;
    uint32_t CountChannelsByScope(bool crossServer) const;
    uint32_t CountChannelsByProtocol(bool crossServer, CommProtocol protocol) const;
    uint32_t CountRecognizedChannelsByScope(bool crossServer) const;
    HcclResult NotifyReadyByScopeAndProtocol(bool crossServer, CommProtocol protocol) const;
    HcclResult ReadRemoteRanksByScopeAndProtocol(bool crossServer, CommProtocol protocol) const;
    HcclResult RunScope(bool crossServer) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    uint64_t packedBytes_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
