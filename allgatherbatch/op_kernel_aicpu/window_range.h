#ifndef HCCL_ALLGATHERBATCH_WINDOW_RANGE_H
#define HCCL_ALLGATHERBATCH_WINDOW_RANGE_H

#include <cstdint>

namespace ops_hccl_allgatherbatch {

struct WindowRange {
    // 当前窗口从哪个 item 开始，以及在该 item 内从哪个字节偏移开始。
    uint32_t startItemIdx = 0;
    uint64_t startOffsetBytes = 0;

    // 当前窗口覆盖到哪个 item 结束。endOffsetBytes 的精确语义在阶段 5 再细化。
    uint32_t endItemIdx = 0;
    uint64_t endOffsetBytes = 0;

    // 这个窗口最终会被 Pack 成多少连续字节。
    uint64_t packedBytes = 0;
};

}  // namespace ops_hccl_allgatherbatch

#endif
