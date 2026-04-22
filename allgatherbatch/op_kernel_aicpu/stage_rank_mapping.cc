#include "stage_rank_mapping.h"

#include <algorithm>

namespace ops_hccl_allgatherbatch {

namespace {

static void ReorderSequence(u32 start, u32 end, u32 len, std::vector<u32> &tree, std::vector<u32> &tmp)
{
    for (u32 i = start; i < end; ++i) {
        const u32 offset = i - start;
        if ((offset & 1U) == 0U) {
            tmp[start + offset / 2U] = tree[i];
        } else {
            tmp[start + (offset + len) / 2U] = tree[i];
        }
    }
}

}  // namespace

u32 GetStepNumInterServer(u32 rankSize)
{
    if (rankSize <= 1U) {
        return 0U;
    }

    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1U; tmp != 0U; tmp >>= 1U) {
        ++nSteps;
    }
    return nSteps;
}

void GetRankMapping(u32 rankSize, std::vector<u32> &sliceMap, bool keepOrder)
{
    sliceMap.clear();
    sliceMap.resize(rankSize, 0U);
    if (keepOrder || rankSize <= 1U) {
        for (u32 i = 0; i < rankSize; ++i) {
            sliceMap[i] = i;
        }
        return;
    }

    std::vector<u32> tree;
    tree.reserve(rankSize);
    for (u32 i = 0; i < rankSize; ++i) {
        tree.push_back(i);
    }

    std::vector<u32> tmp(rankSize, 0U);
    const u32 nSteps = GetStepNumInterServer(rankSize);
    u32 len = rankSize;
    for (u32 step = 0; step < nSteps; ++step) {
        const u32 nSlices = (rankSize - 1U + (1U << step)) / (1U << (step + 1U));
        if (nSlices <= 1U) {
            break;
        }

        bool endFlag = false;
        for (u32 part = 0; part * len < rankSize; ++part) {
            const u32 start = part * len;
            const u32 end = std::min(start + len, rankSize);
            ReorderSequence(start, end, len, tree, tmp);
            if (((end - start) & 1U) == 1U) {
                endFlag = true;
            }
        }

        tree.swap(tmp);
        if (endFlag) {
            break;
        }
        len >>= 1U;
    }

    for (u32 i = 0; i < rankSize; ++i) {
        sliceMap[tree[i]] = i;
    }
}

}  // namespace ops_hccl_allgatherbatch
