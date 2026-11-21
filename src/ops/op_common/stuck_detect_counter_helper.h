/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUCK_DETECT_COUNTER_HELPER_H
#define STUCK_DETECT_COUNTER_HELPER_H

#include <chrono>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "hccl_aiv_utils.h"
#include "hccl_host_comm_dl.h"
#include "hccl_res_dl.h"
#include "ccu_res_dl.h"
#include "hcomm_dlsym.h"
#include "log.h"

namespace ops_hccl {

constexpr const char* STUCK_DETECT_COUNTER_CACHE_NAME = "stuck_detect_counter_cache";
constexpr uint64_t STUCK_DETECT_CACHE_REQUERY_MS = 1000;
constexpr uint64_t STUCK_DETECT_CACHE_EXPIRE_MS = 10 * 60 * 1000;
constexpr size_t STUCK_DETECT_CACHE_MAX_ENTRIES = 256;
constexpr uint64_t STUCK_DETECT_COUNTER_SLOT_BYTES = 16;

struct StuckDetectCounterEntry {
    OpCounterInfo counter = {};
    uint64_t token = 0;
    bool tokenQueried = false;
    bool isQuerySucceeded = false;
    std::string commName;
    uint32_t rank = 0;
    uint64_t lastQueryMs = 0;
};

class StuckDetectCounterCache {
public:
    static StuckDetectCounterCache& GetInstance()
    {
        static StuckDetectCounterCache cache;
        return cache;
    }

    static uint64_t NowMs()
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    bool Get(HcclComm comm, StuckDetectCounterEntry& entry, uint64_t& generation) const
    {
        StuckDetectCounterEntry cached;
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            generation = generation_;
            auto iter = entries_.find(comm);
            if (iter == entries_.end()) {
                return false;
            }
            cached = iter->second;
        }

        if (NowMs() - cached.lastQueryMs >= STUCK_DETECT_CACHE_REQUERY_MS) {
            HCCL_DEBUG("[StuckDetect][Cache] entry expired, comm[%p].", comm);
            return false;
        }
        if (!IsSameComm(comm, cached)) {
            HCCL_INFO(
                "[StuckDetect][Cache] identity mismatch, comm[%p], cachedName[%s], cachedRank[%u].", comm,
                cached.commName.c_str(), cached.rank);
            return false;
        }
        entry = cached;
        return true;
    }

    bool Put(HcclComm comm, const StuckDetectCounterEntry& entry, uint64_t generation, bool& isFirstInsert)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (generation_ != generation) {
            HCCL_DEBUG(
                "[StuckDetect][Cache] drop stale put, comm[%p], generation[%llu]->[%llu].", comm, generation,
                generation_);
            return false;
        }
        const uint64_t nowMs = NowMs();
        ClearExpiredLocked(nowMs);
        isFirstInsert = (entries_.find(comm) == entries_.end());
        if (isFirstInsert && entries_.size() >= STUCK_DETECT_CACHE_MAX_ENTRIES) {
            EvictOldestLocked();
        }
        entries_[comm] = entry;
        return true;
    }

    void Erase(HcclComm comm)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        ++generation_;
        entries_.erase(comm);
    }

    size_t Size() const
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return entries_.size();
    }

private:
    StuckDetectCounterCache() = default;
    ~StuckDetectCounterCache() = default;
    StuckDetectCounterCache(const StuckDetectCounterCache&) = delete;
    StuckDetectCounterCache& operator=(const StuckDetectCounterCache&) = delete;
    StuckDetectCounterCache(StuckDetectCounterCache&&) = delete;
    StuckDetectCounterCache& operator=(StuckDetectCounterCache&&) = delete;

    static bool IsSameComm(HcclComm comm, const StuckDetectCounterEntry& cached)
    {
        char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
        uint32_t rank = 0;
        if (HcclGetCommName(comm, commName) != HCCL_SUCCESS || HcclGetRankId(comm, &rank) != HCCL_SUCCESS) {
            return false;
        }
        return cached.commName == commName && cached.rank == rank;
    }

    void ClearExpiredLocked(uint64_t nowMs)
    {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (nowMs - it->second.lastQueryMs > STUCK_DETECT_CACHE_EXPIRE_MS) {
                HCCL_DEBUG("[StuckDetect][Cache] drop idle entry, comm[%p].", it->first);
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void EvictOldestLocked()
    {
        auto oldest = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.lastQueryMs < oldest->second.lastQueryMs) {
                oldest = it;
            }
        }
        HCCL_DEBUG("[StuckDetect][Cache] evict oldest, comm[%p], size[%zu].", oldest->first, entries_.size());
        entries_.erase(oldest);
    }

    mutable std::shared_mutex mutex_;
    std::unordered_map<HcclComm, StuckDetectCounterEntry> entries_;
    uint64_t generation_ = 0; // 失效代次，受 mutex_ 保护
};

// ---------- 缓存失效回调 ----------
inline HcclResult StuckDetectCounterCacheStateCallback(HcclComm comm, HcclCommStatePhase state, void* args)
{
    (void)args;
    if (comm == nullptr) {
        return HCCL_SUCCESS;
    }
    if (state != HCCL_COMM_STATE_PHASE_DESTROY_PRE && state != HCCL_COMM_STATE_PHASE_DESTROY_POST
        && state != HCCL_COMM_STATE_PHASE_RESUME_POST) {
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[StuckDetect][CacheCb] erase comm[%p], state[%d].", comm, static_cast<int>(state));
    StuckDetectCounterCache::GetInstance().Erase(comm);
    return HCCL_SUCCESS;
}

inline bool RegisterStuckDetectCounterCacheCallback()
{
    static std::once_flag once;
    static bool isRegistered = false;
    std::call_once(once, []() {
        HcommDlInit();
        if (!HcommIsSupportHcclCommRegCommStateCallback()) {
            HCCL_INFO("[StuckDetect][CacheCb] end, registered[0] (unsupported).");
            return;
        }
        static u32 args = 0;
        HcclResult ret = HcclCommRegCommStateCallback(
            STUCK_DETECT_COUNTER_CACHE_NAME, StuckDetectCounterCacheStateCallback, &args);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[StuckDetect][CacheCb] register failed, ret[%d].", ret);
            return;
        }
        isRegistered = true;
        HCCL_INFO("[StuckDetect][CacheCb] end, registered[1].");
    });
    return isRegistered;
}

// ---------- 对外辅助接口 ----------

// 记录本次查询的通信域身份；取不到身份时返回 false，调用方据此不写缓存
inline bool FillStuckDetectCounterIdentity(HcclComm comm, StuckDetectCounterEntry& entry)
{
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    uint32_t rank = 0;
    if (HcclGetCommName(comm, commName) != HCCL_SUCCESS || HcclGetRankId(comm, &rank) != HCCL_SUCCESS) {
        HCCL_DEBUG("[StuckDetect][Query] no identity, skip cache, comm[%p].", comm);
        return false;
    }
    entry.commName = commName;
    entry.rank = rank;
    return true;
}

inline HcclResult GetStuckDetectCounterInfo(HcclComm comm, OpCounterInfo& counter, uint64_t* token = nullptr)
{
    if (comm == nullptr) {
        counter = {};
        if (token != nullptr) {
            *token = 0;
        }
        return HCCL_SUCCESS;
    }
    HcommDlInit(); // 幂等：保证能力位已探测（静态库构建下不依赖 compat 构造器）
    if (!HcommIsSupportHcclStuckDetectGetCounterAddr()) {
        HCCL_DEBUG("[StuckDetect][Query] GetCounterAddr unsupported, skip.");
        counter = {};
        if (token != nullptr) {
            *token = 0;
        }
        return HCCL_SUCCESS;
    }

    StuckDetectCounterEntry entry;
    uint64_t generation = 0;
    // 回调能力不支持/注册失败时不启用缓存，每次直查（与旧 hcomm 兼容）
    const bool isCacheEnabled = RegisterStuckDetectCounterCacheCallback();
    bool isCacheHit = isCacheEnabled && StuckDetectCounterCache::GetInstance().Get(comm, entry, generation);
    // 非 CCU 调用方不查询 token；CCU 调用方命中这类条目时按 miss 重查，
    // 避免缓存中的 token=0 让 CCU 计数被误关（token 只随 CCU 调用方的查询写入）
    if (isCacheHit && token != nullptr && !entry.tokenQueried) {
        HCCL_DEBUG("[StuckDetect][Query] token not cached, requery, comm[%p].", comm);
        isCacheHit = false;
    }
    if (!isCacheHit) {
        entry = {}; // 未命中/身份不一致/已过期/缺 token：重新取址
        uint64_t headAddr = 0;
        uint64_t tailAddr = 0;
        uint64_t addOneAddr = 0;
        HcclResult ret = HcclStuckDetectGetCounterAddr(comm, &headAddr, &tailAddr, &addOneAddr);
        if (ret != HCCL_SUCCESS) {
            HCCL_DEBUG("[StuckDetect][Query] get addr failed, comm[%p], ret[%d].", comm, static_cast<int>(ret));
        }
        if (ret == HCCL_SUCCESS) {
            entry.isQuerySucceeded = true;
            if (headAddr != 0 && tailAddr != 0 && addOneAddr != 0) { // 三地址齐全才算使能
                entry.counter.headCountMem = headAddr;
                entry.counter.tailCountMem = tailAddr;
                entry.counter.addOneMem = addOneAddr;
                entry.counter.memSize = static_cast<u32>(sizeof(int32_t));
                entry.counter.isEnableCounter = true;
                // 计数槽 token 仅 CCU 调用方需要：按 (slot 基址, 16B) 查询；
                // 失败归一为 0，由 CCU 侧关闭计数，非 CCU 路径不承担该控制面查询与告警
                if (token != nullptr) {
                    CcuResult tokenRet = HcommCcuGetMemToken(headAddr, STUCK_DETECT_COUNTER_SLOT_BYTES, &entry.token);
                    if (tokenRet != CCU_SUCCESS) {
                        HCCL_WARNING(
                            "[StuckDetect][Query] get CCU counter token failed, comm[%p], ret[%d].", comm,
                            static_cast<int>(tokenRet));
                        entry.token = 0;
                    }
                    // 失败也标记已查询：按 1s 时效重试，避免每次调用重复查询
                    entry.tokenQueried = true;
                }
            } else if (token != nullptr) {
                // 计数未使能：无计数槽可查 token，标记后避免 CCU 命中时反复按 miss 重查
                entry.tokenQueried = true;
            }
        }
        if (isCacheEnabled && entry.isQuerySucceeded && FillStuckDetectCounterIdentity(comm, entry)) {
            entry.lastQueryMs = StuckDetectCounterCache::NowMs();
            bool isFirstInsert = false;
            const bool isPut = StuckDetectCounterCache::GetInstance().Put(comm, entry, generation, isFirstInsert);
            if (isPut && isFirstInsert) { // 仅首次入库打印，避免 TTL 重查刷屏
                if (entry.counter.isEnableCounter) {
                    HCCL_INFO(
                        "[StuckDetect][Query] comm[%p] enabled(head[%p], tail[%p]).", comm,
                        reinterpret_cast<void*>(entry.counter.headCountMem),
                        reinterpret_cast<void*>(entry.counter.tailCountMem));
                } else {
                    HCCL_INFO("[StuckDetect][Query] comm[%p] disabled(zero addr).", comm);
                }
            }
        }
    }
    counter = entry.counter;
    if (token != nullptr) {
        *token = entry.counter.isEnableCounter ? entry.token : 0;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

#endif // STUCK_DETECT_COUNTER_HELPER_H
