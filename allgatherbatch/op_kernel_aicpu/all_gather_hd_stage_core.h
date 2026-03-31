#ifndef HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H
#define HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H

#include "common.h"

namespace ops_hccl_allgatherbatch {

class AllGatherHDStageCore {
public:
    AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx);

    HcclResult RunAsync();

private:
    const OpParam &param_;
    AlgResourceCtx &resCtx_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
