/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context arg header file
 * Create: 2025-02-13
 */

#ifndef HCOMM_CCU_TASK_PARAM_H
#define HCOMM_CCU_TASK_PARAM_H

#include <cstdint>

namespace hcomm {

constexpr uint32_t CCU_SQE_ARGS_LEN = 13;

struct CcuTaskParam {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t timeout;
    uint32_t instStartId;
    uint32_t instCnt;
    uint32_t key;
    uint32_t argSize;
    uint64_t args[CCU_SQE_ARGS_LEN];
};

}; // namespace hcomm

#endif // _CCU_TASK_PARAM_H