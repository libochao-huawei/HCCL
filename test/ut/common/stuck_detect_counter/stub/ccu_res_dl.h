/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_CCU_RES_DL_H
#define UT_STUCK_DETECT_CCU_RES_DL_H

#include <cstdint>

// 最小桩：helper 的 CCU token 查询接口；固定 token 由 stuck_stub.cc 提供。
using CcuResult = int;
constexpr CcuResult CCU_SUCCESS = 0;

extern "C" CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t* tokenInfo);

#endif // UT_STUCK_DETECT_CCU_RES_DL_H
