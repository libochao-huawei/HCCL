/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <cstring>
#include <acl/acl_rt.h>
#include "utils.h"

namespace {
constexpr std::unordered_map<std::string, DeviceType> SOC_VERSION_MAP{
    {"Ascend910B1", DEVICE_TYPE_A2},
    {"Ascend910B2", DEVICE_TYPE_A2},
    {"Ascend910B2C", DEVICE_TYPE_A2},
    {"Ascend910B3", DEVICE_TYPE_A2},
    {"Ascend910B4", DEVICE_TYPE_A2},
    {"Ascend910B4-1", DEVICE_TYPE_A2},

    {"Ascend910_9391", DEVICE_TYPE_A3},
    {"Ascend910_9381", DEVICE_TYPE_A3},
    {"Ascend910_9392", DEVICE_TYPE_A3},   // Ascend910_9392、Ascend910_9382为预留类型，当前版本暂不支持，待跟随后续版本节奏交付
    {"Ascend910_9382", DEVICE_TYPE_A3},
    {"Ascend910_9372", DEVICE_TYPE_A3},
    {"Ascend910_9362", DEVICE_TYPE_A3},


    {"Ascend910B1", DevType::DEV_TYPE_910A2},
    {"Ascend910B2", DevType::DEV_TYPE_910A2},
    {"Ascend910B3", DevType::DEV_TYPE_910A2},
    {"Ascend910B4", DevType::DEV_TYPE_910A2},
    {"Ascend910_939", DevType::DEV_TYPE_910A3},
    {"Ascend910_938", DevType::DEV_TYPE_910A3},
    {"Ascend910_937", DevType::DEV_TYPE_910A3},
}
}

namespace ops_hccl_p2p {
DeviceType GetDeviceType() {
    const char *socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr,
                HCCL_ERROR("[GetDeviceType] Failed to get soc name"),
                HCCL_E_RUNTIME);

    std::string socName(socNamePtr);
    if (socName.find("Ascend910B") != std::string::npos) {
        return DEVICE_TYPE_A2;
    } else if (socName.find("Ascend910_93") != std::string::npos) {
        return DEVICE_TYPE_A3;
    } else if (socName.find("Ascend950") != std::string::npos) {
        return DEVICE_TYPE_A5;
    } else {

    }
}
}
