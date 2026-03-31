#ifndef HCCL_ALLGATHERBATCH_WINDOW_RANGE_H
#define HCCL_ALLGATHERBATCH_WINDOW_RANGE_H

#include <cstdint>

namespace ops_hccl_allgatherbatch {

struct WindowRange {
    uint32_t startItemIdx = 0;
    uint64_t startOffsetBytes = 0;
    uint32_t endItemIdx = 0;
    uint64_t endOffsetBytes = 0;
    uint64_t packedBytes = 0;
};

}  // namespace ops_hccl_allgatherbatch

#endif
