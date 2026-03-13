/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 公共变量
 * Author: wenxuemin
 * Create: 2024-02-04
 */
#ifndef HCCLV2_CONST_VAL_H
#define HCCLV2_CONST_VAL_H
// #include "types.h"
namespace Hccl {
constexpr u32 CANN_VERSION_MAX_LEN = 50; // Cann版本信息的最大长度为50
constexpr u32 SOC_VERSION_MAX_LEN  = 32; // soc version的最大长度为32
constexpr u32 INVALID_U32 = UINT32_MAX;
constexpr u64 INVALID_U64 = UINT64_MAX;
constexpr s32 INVALID_RANKID = INT32_MAX;
} // namespace Hccl
#endif