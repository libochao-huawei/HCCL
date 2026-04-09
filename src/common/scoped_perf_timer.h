/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SCOPED_PERF_TIMER_H
#define HCCL_SCOPED_PERF_TIMER_H

#include <cstdlib>
#include <string>
#include <strings.h>
#include "log.h"
#include "sal.h"

namespace ops_hccl {
inline bool IsScopedPerfEnvEnabled(const char *envName)
{
    if (envName == nullptr) {
        return true;
    }
    const char *envValue = std::getenv(envName);
    if (envValue == nullptr) {
        return false;
    }
    return strcmp(envValue, "0") != 0 && strcasecmp(envValue, "false") != 0 &&
        strcasecmp(envValue, "off") != 0 && strcasecmp(envValue, "no") != 0;
}

class HcclScopedPerfTimer {
public:
    HcclScopedPerfTimer(const char *scopeName, bool enabled = true, u64 slowThresholdUs = 0,
        bool logAsError = false)
        : scopeName_(scopeName == nullptr ? "NA" : scopeName),
          enabled_(enabled),
          slowThresholdUs_(slowThresholdUs),
          logAsError_(logAsError)
    {
        if (enabled_) {
            startUs_ = TIME_NOW();
        }
    }

    HcclScopedPerfTimer(const char *scopeName, const char *envName, u64 slowThresholdUs = 0,
        bool logAsError = false)
        : HcclScopedPerfTimer(scopeName, IsScopedPerfEnvEnabled(envName), slowThresholdUs, logAsError)
    {
    }

    ~HcclScopedPerfTimer()
    {
        if (!enabled_) {
            return;
        }
        const u64 totalUs = static_cast<u64>(DURATION_US(TIME_NOW() - startUs_).count());
        if (slowThresholdUs_ > 0 && totalUs < slowThresholdUs_) {
            return;
        }

        if (logAsError_) {
            HCCL_ERROR("[ScopedPerf] scope[%s], total[%llu]us%s%s",
                scopeName_.c_str(),
                static_cast<unsigned long long>(totalUs),
                extra_.empty() ? "" : ", ",
                extra_.empty() ? "" : extra_.c_str());
        } else {
            HCCL_INFO("[ScopedPerf] scope[%s], total[%llu]us%s%s",
                scopeName_.c_str(),
                static_cast<unsigned long long>(totalUs),
                extra_.empty() ? "" : ", ",
                extra_.empty() ? "" : extra_.c_str());
        }
    }

    void SetExtra(const std::string &extra)
    {
        extra_ = extra;
    }

    void AppendExtra(const std::string &extra)
    {
        if (extra.empty()) {
            return;
        }
        if (!extra_.empty()) {
            extra_ += ", ";
        }
        extra_ += extra;
    }

    u64 ElapsedUs() const
    {
        if (!enabled_) {
            return 0;
        }
        return static_cast<u64>(DURATION_US(TIME_NOW() - startUs_).count());
    }

    void Cancel()
    {
        enabled_ = false;
    }

private:
    std::string scopeName_;
    bool enabled_ = false;
    u64 slowThresholdUs_ = 0;
    bool logAsError_ = false;
    HcclUs startUs_{};
    std::string extra_;
};
} // namespace ops_hccl

#define HCCL_SCOPED_PERF(timerVar, scopeName, envName, slowUs) \
    ops_hccl::HcclScopedPerfTimer timerVar(scopeName, envName, slowUs)

#define HCCL_SCOPED_PERF_IF(timerVar, scopeName, enabled, slowUs) \
    ops_hccl::HcclScopedPerfTimer timerVar(scopeName, enabled, slowUs)

#define HCCL_SCOPED_PERF_ERR(timerVar, scopeName, envName, slowUs) \
    ops_hccl::HcclScopedPerfTimer timerVar(scopeName, envName, slowUs, true)

#define HCCL_SCOPED_PERF_ERR_IF(timerVar, scopeName, enabled, slowUs) \
    ops_hccl::HcclScopedPerfTimer timerVar(scopeName, enabled, slowUs, true)

#endif // HCCL_SCOPED_PERF_TIMER_H
