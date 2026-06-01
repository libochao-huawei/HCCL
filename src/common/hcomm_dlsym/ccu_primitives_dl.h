/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CCU_PRIMITIVES_DL_H
#define CCU_PRIMITIVES_DL_H

#include "dlsym_common.h"
#if CANN_VERSION_NUM >= 90100000
#include "ccu_primitives.hpp"   // 原始头文件，包含所有类型和声明
#endif

namespace AscendC {
namespace ccu {

// ==================== 类型别名 ====================
typedef struct {
    uint64_t addrOffset;
    uint64_t loopIterNum;
} CcuLoopConfig;
typedef struct {
    uint64_t addrOffset;
    uint64_t bufferOffset;
    uint64_t eventOffset;
    uint64_t repeatNum;
    uint64_t repeatLoopIdx;
} CcuLoopGroupConfig;
using LoopConfig      = ::CcuLoopConfig;
using LoopGroupConfig = ::CcuLoopGroupConfig;

class Variable;

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(Variable, GetResByChannel, ChannelHandle channel, uint32_t varIndex);

void CcuPrimitivesDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

} // namespace ccu
} // namespace AscendC

#endif //CCU_PRIMITIVES_DL_H
