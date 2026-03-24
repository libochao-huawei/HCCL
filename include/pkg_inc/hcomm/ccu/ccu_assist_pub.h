/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context header file
 * Create: 2025-02-18
 */

#ifndef CCU_CONTEXT_ASSIST_PUB_H
#define CCU_CONTEXT_ASSIST_PUB_H

#include <cstdint>

namespace hcomm {
namespace CcuRep {

// 辅助函数
uint64_t GetTokenInfo(uint64_t va, uint64_t size);

}; // namespace CcuRep
}; // namespace hcomm

#endif // _CCU_CONTEXT_ASSIST_PUB_H