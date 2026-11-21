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
 * @file ccu_stuck_counter.h
 * @brief CCU 卡死检测计数：trampoline 与 kernel 内 mission mark 共用同一份 LocalReduce 实现。
 *
 * Event 申请说明：CcuEventAlloc 属于当前 kernel build，经 GetResourceRequest() 计入 blockCkeReq，
 *   随资源查询写入描述、注册时申请；句柄不能跨 build 复用，故每次 build 首次 Emit 申请 1 个供
 *   head/tail 复用；申请失败仅降级为不计数（catch CcuException 与未知异常，保证 fail-open）。
 */

#ifndef CCU_STUCK_COUNTER_H
#define CCU_STUCK_COUNTER_H

#include <cstdint>
#include <exception>
#include <optional>

#include "alg_param.h"
#include "ccu_primitives_dl.hpp"
#include "log.h"

namespace ops_hccl {

constexpr uint64_t CCU_STUCK_COUNTER_BYTES = sizeof(int32_t);
constexpr uint16_t CCU_STUCK_COUNTER_EVT_MASK = 1;

class CcuStuckCounter {
public:
    explicit CcuStuckCounter(const CcuKernelArgBase* arg) : arg_(arg) {} // 构造不申请 Event，避免无打点时白占资源

    CcuResult Head() { return Emit(arg_ == nullptr ? 0 : arg_->stuckHeadAddr); }

    CcuResult Tail() { return Emit(arg_ == nullptr ? 0 : arg_->stuckTailAddr); }

private:
    bool CanEmit(uint64_t dstAddr) const
    {
        return arg_->stuckCounterEnabled && dstAddr != 0 && arg_->stuckAddOneAddr != 0;
    }

    // 每次 build 最多申请 1 个 Event（head/tail 复用）；任何异常（含未知类型）都只降级不抛出，保证 fail-open
    bool EnsureEvent()
    {
        if (evt_.has_value()) {
            return true;
        }
        if (allocFailed_ || arg_ == nullptr || !(CanEmit(arg_->stuckHeadAddr) || CanEmit(arg_->stuckTailAddr))) {
            return false;
        }
        try {
            evt_.emplace();
        } catch (const std::exception& err) {
            allocFailed_ = true;
            HCCL_DEBUG("[StuckDetect][CcuEmit] event alloc failed, skip counting, why[%s].", err.what());
            return false;
        } catch (...) {
            allocFailed_ = true;
            HCCL_DEBUG("[StuckDetect][CcuEmit] event alloc failed(unknown), skip counting.");
            return false;
        }
        return true;
    }

    CcuResult Emit(uint64_t dstAddr)
    {
        if (broken_ || arg_ == nullptr || !CanEmit(dstAddr) || !EnsureEvent()) {
            return CCU_SUCCESS;
        }
        bool counted = false;
        try {
            hcomm::ccu::LocalAddr dst;
            dst.addr = dstAddr;
            dst.token = arg_->stuckCounterToken;
            hcomm::ccu::LocalAddr src;
            src.addr = arg_->stuckAddOneAddr;
            src.token = arg_->stuckCounterToken;
            hcomm::ccu::Variable len;
            len = CCU_STUCK_COUNTER_BYTES;
            CcuResult ret = hcomm::ccu::LocalReduce(
                dst, src, len, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM, *evt_,
                CCU_STUCK_COUNTER_EVT_MASK);
            if (ret != CCU_SUCCESS) {
                broken_ = true;
                return ret;
            }
            counted = true;
            return hcomm::ccu::EventWait(*evt_, CCU_STUCK_COUNTER_EVT_MASK);
        } catch (const std::exception& err) {
            broken_ = !counted;
            HCCL_WARNING(
                "[StuckDetect][CcuEmit] exception, skip counting, enabled[%d], why[%s].", arg_->stuckCounterEnabled,
                err.what());
            return CCU_SUCCESS;
        } catch (...) {
            broken_ = !counted;
            HCCL_WARNING(
                "[StuckDetect][CcuEmit] exception(unknown), skip counting, enabled[%d].", arg_->stuckCounterEnabled);
            return CCU_SUCCESS;
        }
    }

    const CcuKernelArgBase* arg_{nullptr};
    std::optional<hcomm::ccu::Event> evt_;
    bool allocFailed_{false};
    bool broken_{false};
};

} // namespace ops_hccl

#endif // CCU_STUCK_COUNTER_H
