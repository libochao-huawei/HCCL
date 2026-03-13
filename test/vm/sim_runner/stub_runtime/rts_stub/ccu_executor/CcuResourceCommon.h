/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor resource manager
 * Author: z00445483
 */

#ifndef HCCL_SIM_CCU_RESOURCE_COMMON_H
#define HCCL_SIM_CCU_RESOURCE_COMMON_H

using namespace std;

constexpr int DIE_NUM = 2;
constexpr uint32_t BYTE_NUM_4K                  = 4096;

struct CcuInfo {
    int rankId{INT32_MAX};
    int dieId{INT32_MAX};
};

#endif // HCCL_SIM_CCU_RESOURCE_COMMON_H
