/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CONFIG_H
#define HCCL_CONFIG_H

#include <stdint.h>

#if !defined(HCCL_OP_EXPANSION_MODE_AI_CPU)
/**
 * @enum HcclOpExpansionMode
 * @brief Operator expansion mode exposed by HCCL.
 */
typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID = -1,
    HCCL_OP_EXPANSION_MODE_AI_CPU = 0,
    HCCL_OP_EXPANSION_MODE_AIV = 1,
    HCCL_OP_EXPANSION_MODE_HOST = 2,
    HCCL_OP_EXPANSION_MODE_HOST_TS = 3,
} HcclOpExpansionMode;
#endif

#if !defined(HCCL_CONFIG_TYPE_OP_EXPANSION_MODE)
/**
 * @enum HcclConfigType
 * @brief Configuration item selectors exposed by HCCL.
 */
typedef enum {
    HCCL_CONFIG_TYPE_INVALID = -1,
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE = 0,
} HcclConfigType;
#endif

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclConfigGetInfo(HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, void *info);

#ifdef __cplusplus
}
#endif

#endif  // HCCL_CONFIG_H
