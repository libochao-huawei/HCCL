#ifndef HCCL_ALLGATHERBATCH_WINDOW_RANGE_H
#define HCCL_ALLGATHERBATCH_WINDOW_RANGE_H

#include <cstdint>

namespace ops_hccl_allgatherbatch {

struct WindowRange {
    // 当前窗口从哪个 item 开始，以及在该 item 内从哪个字节偏移开始。
    uint32_t startItemIdx = 0;
    uint64_t startOffsetBytes = 0;

    // 当前窗口的结束位置采用“尾后游标”语义：
    // 1. 如果 endItemIdx < itemCount，则 [endItemIdx, endOffsetBytes) 是下一个未处理位置。
    // 2. 如果 endItemIdx == itemCount，则表示窗口正好覆盖到了全部输入末尾，此时 endOffsetBytes 应为 0。
    uint32_t endItemIdx = 0;
    uint64_t endOffsetBytes = 0;

    // 这个窗口最终会被 Pack 成多少连续字节。
    uint64_t packedBytes = 0;
};

}  // namespace ops_hccl_allgatherbatch

#endif
