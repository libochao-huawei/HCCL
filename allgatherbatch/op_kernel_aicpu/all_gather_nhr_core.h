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
    AllGatherNHRCore(const OpParam &param, AlgResourceCtx &resCtx);

    // NHR 子模板入口：先把跨 rank 的步进关系算出来，真正的数据搬运留到后续阶段接入。
    HcclResult RunAsync();

private:
    HcclResult ValidateCommState() const;
    uint32_t CalcStepNum(uint32_t rankSize) const;
    HcclResult GetStepInfo(uint32_t step, uint32_t nSteps, NHRStepInfo &stepInfo) const;
    HcclResult BuildStepPlan(std::vector<NHRStepInfo> &stepPlan) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
