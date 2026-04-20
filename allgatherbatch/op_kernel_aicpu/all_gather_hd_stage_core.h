#ifndef HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H
#define HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H

#include <vector>

#include "common.h"

namespace ops_hccl_allgatherbatch {

class AllGatherHDStage {
public:
    AllGatherHDStage(const OpParam &param, AlgResourceCtx &resCtx, ExecMem &execMem, std::vector<ChannelResource> &channels);
    HcclResult RunAsync();

private:
    HcclResult ReverseId(u32 oriIdx, u32 &revIdx);
    HcclResult RunPreCopy();
    HcclResult RunAllGatherNoPower();
    HcclResult PrepareSliceData(u32 subRank, u32 subRankSize, u32 size, u32 batchSize, std::vector<Slice> &slices);
    HcclResult MainRecordSub(u32 threadNum);
    HcclResult SubWaitMain(u32 threadNum);
    HcclResult MainWaitSub(u32 threadNum);
    HcclResult SubRecordMain(u32 threadNum);
    HcclResult RunBetweenStep(u32 neighCur, u32 neighNext);
    HcclResult RunAllGatherPower();
    HcclResult RunAllGatherLastTwo();
    HcclResult RunAllGatherLastOne();
    HcclResult RunAllGatherLast();
    HcclResult RunAllGatherStage();

    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }
    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    ExecMem &execMem_;
    std::vector<ChannelResource> &channels_;
    std::vector<Slice> sliceNoPower_;
    std::vector<Slice> slicePower_;
    u32 rank{0};
    u32 rankSize{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_RESERVED};
    
    const std::vector<u32> resMap = {0, 2, 1, 3};
    u32 totalSize_{0};
    const u32 base = 2;
    u32 nSteps_ = 0;
    u32 powerSteps_ = 0;
    u32 finalSteps_ = 0;
    u32 noPower_ = 0;
};

}  // namespace ops_hccl_allgatherbatch

#endif