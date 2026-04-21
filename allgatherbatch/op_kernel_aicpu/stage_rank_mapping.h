#ifndef HCCL_ALLGATHERBATCH_STAGE_RANK_MAPPING_H
#define HCCL_ALLGATHERBATCH_STAGE_RANK_MAPPING_H

#include <vector>

#include "common.h"

namespace ops_hccl_allgatherbatch {

u32 GetStepNumInterServer(u32 rankSize);
void GetRankMapping(u32 rankSize, std::vector<u32> &sliceMap, bool keepOrder = false);

}  // namespace ops_hccl_allgatherbatch

#endif
