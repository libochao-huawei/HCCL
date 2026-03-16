#ifndef OPS_HCCL_CUSTOM_EXTRA_ARGS_H
#define OPS_HCCL_CUSTOM_EXTRA_ARGS_H

#include <cstdint>

constexpr uint32_t MAX_RANK_SIZE = 8;

struct ExtraArgs {
    uint64_t sendCounts[MAX_RANK_SIZE] = {};
    uint64_t sendDispls[MAX_RANK_SIZE] = {};
    uint64_t recvCounts[MAX_RANK_SIZE] = {};
    uint64_t recvDispls[MAX_RANK_SIZE] = {};
};

#endif // OPS_HCCL_CUSTOM_EXTRA_ARGS_H
