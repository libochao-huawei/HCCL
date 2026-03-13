#ifndef HCCL_VM_TYPE_H
#define HCCL_VM_TYPE_H

enum HcclSimMode {
    CHECKER,
    RUNNER
};

constexpr size_t MAX_INPUT_LEN = 4096;
constexpr size_t MAX_ARGC_NUM = 1000;
constexpr size_t MAX_SUBCMD_LEN = 15;
#endif