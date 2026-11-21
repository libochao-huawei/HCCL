/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aicpu_stuck_counter.h
 * @brief AICPU 卡死检测计数下发：以同流 inline reduce SQE 对 head/tail 执行 counter += addOne。
 */

#ifndef AICPU_STUCK_COUNTER_H
#define AICPU_STUCK_COUNTER_H

#include <cstdint>

#include "alg_param.h"
#include "hcomm_primitives.h"
#include "log.h"

namespace ops_hccl {

// 计数语义：head 下发前、tail 返回后发射（含排队等待）；三地址齐全才发射，任一为 0 两侧都不发
inline bool AicpuStuckCounterReady(ThreadHandle thread, const OpParam& param)
{
    return param.isEnableCounter != 0 && thread != 0 && param.counterHeadAddr != 0 && param.counterTailAddr != 0
           && param.counterAddOneAddr != 0;
}

inline bool AicpuStuckCounterEmit(ThreadHandle thread, const OpParam& param, bool isHead)
{
    if (!AicpuStuckCounterReady(thread, param)) {
        HCCL_DEBUG(
            "[StuckDetect][AicpuEmit] skip, thread[%llu], head[0x%llx], tail[0x%llx], addOne[0x%llx].", thread,
            param.counterHeadAddr, param.counterTailAddr, param.counterAddOneAddr);
        return false;
    }
    uint64_t countAddr = isHead ? param.counterHeadAddr : param.counterTailAddr;
    int32_t ret = HcommLocalReduceOnThread(
        thread, reinterpret_cast<void*>(countAddr), reinterpret_cast<const void*>(param.counterAddOneAddr), 1,
        HCOMM_DATA_TYPE_INT32, HCOMM_REDUCE_SUM);
    if (ret != 0) {
        HCCL_DEBUG("[StuckDetect][AicpuEmit] LocalReduce failed, thread[%llu], dst[0x%llx].", thread, countAddr);
        return false; // head 未真正下发时调用方会跳过 tail，避免计数器单向偏移
    }
    return true;
}

inline bool AicpuStuckCounterHead(ThreadHandle thread, const OpParam& param)
{
    return AicpuStuckCounterEmit(thread, param, true);
}

inline bool AicpuStuckCounterTail(ThreadHandle thread, const OpParam& param)
{
    if (AicpuStuckCounterEmit(thread, param, false)) {
        return true;
    }
    if (!AicpuStuckCounterReady(thread, param)) {
        return false;
    }
    HCCL_WARNING("[StuckDetect][AicpuEmit] tail LocalReduce failed, retry once, thread[%llu].", thread);
    return AicpuStuckCounterEmit(thread, param, false);
}

inline bool AicpuStuckCounterNeedInvalidateCache(const OpParam& param, bool headEmitted, bool tailEmitted)
{
    return param.isEnableCounter != 0 && (!headEmitted || !tailEmitted);
}

} // namespace ops_hccl

#endif // AICPU_STUCK_COUNTER_H
