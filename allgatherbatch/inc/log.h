#ifndef HCCL_ALLGATHERBATCH_LOG_H
#define HCCL_ALLGATHERBATCH_LOG_H

#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>

#include <dlog_pub.h>
#include "acl/acl.h"
#include "hccl/base.h"
#include "hccl/hccl_types.h"

#ifndef LIKELY
#define LIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 1)))
#define UNLIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 0)))
#endif

#ifndef RUN_LOG_MASK
#define RUN_LOG_MASK 0
#endif

enum LogLevel {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_NONE = 4,
};

enum class HcclSubModuleID {
    LOG_SUB_MODULE_ID_HCCL = 0,
    LOG_SUB_MODULE_ID_HCOM = 1,
    LOG_SUB_MODULE_ID_CLTM = 2,
    LOG_SUB_MODULE_ID_CUSTOM_OP = 3,
};

const u64 SYSTEM_RESERVE_ERROR = 0;
const u64 HCCL_MODULE_ID = 5;

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_ERROR
#endif

inline int32_t HcclAllGatherBatchNormalizeLogLevel(int32_t logType)
{
    if (logType == LOG_LEVEL_DEBUG || logType == static_cast<int32_t>(DLOG_DEBUG)) {
        return LOG_LEVEL_DEBUG;
    }
    if (logType == LOG_LEVEL_INFO || logType == static_cast<int32_t>(DLOG_INFO)) {
        return LOG_LEVEL_INFO;
    }
    if (logType == LOG_LEVEL_WARNING || logType == static_cast<int32_t>(DLOG_WARN)) {
        return LOG_LEVEL_WARNING;
    }
    if (logType == LOG_LEVEL_ERROR || logType == static_cast<int32_t>(DLOG_ERROR)) {
        return LOG_LEVEL_ERROR;
    }
    return LOG_LEVEL_NONE;
}

inline bool &HcclAllGatherBatchErrToWarnFlag()
{
    static bool flag = false;
    return flag;
}

inline bool HcclCheckLogLevel(int logType, int moduleId = static_cast<int>(HCCL))
{
    (void)moduleId;
    return HcclAllGatherBatchNormalizeLogLevel(logType) >=
        HcclAllGatherBatchNormalizeLogLevel(LOG_LEVEL);
}

inline void SetErrToWarnSwitch(bool flag)
{
    HcclAllGatherBatchErrToWarnFlag() = flag;
}

inline bool IsErrorToWarn()
{
    return HcclAllGatherBatchErrToWarnFlag();
}

#define HCCL_LOG_DEBUG DLOG_DEBUG
#define HCCL_LOG_INFO DLOG_INFO
#define HCCL_LOG_WARN DLOG_WARN
#define HCCL_LOG_ERROR DLOG_ERROR
#define HCCL_LOG_MASK (static_cast<int32_t>(HCCL) | static_cast<int32_t>(RUN_LOG_MASK))

#define LOG_FUNC(module, level, fmt, ...) do { \
    DlogRecord(module, level, fmt, ##__VA_ARGS__); \
} while (0)

#define HCCL_LOG_PRINT(moduleId, logType, format, ...) do { \
    LOG_FUNC(moduleId, logType, "[%s:%d] [%u]" format, \
        __FILE__, __LINE__, static_cast<unsigned int>(syscall(SYS_gettid)), ##__VA_ARGS__); \
} while (0)

#define HCCL_ERROR_LOG_PRINT(format, ...) do { \
    if (IsErrorToWarn()) { \
        LOG_FUNC(HCCL_LOG_MASK, HCCL_LOG_WARN, "[%s:%d] [%u]ErrToWarn: " format, \
            __FILE__, __LINE__, static_cast<unsigned int>(syscall(SYS_gettid)), ##__VA_ARGS__); \
    } else { \
        LOG_FUNC(HCCL, HCCL_LOG_ERROR, "[%s:%d] [%u]" format, \
            __FILE__, __LINE__, static_cast<unsigned int>(syscall(SYS_gettid)), ##__VA_ARGS__); \
    } \
} while (0)

#define HCCL_RUN_LOG_PRINT(format, ...) do { \
    LOG_FUNC(HCCL_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u]" format, \
        __FILE__, __LINE__, static_cast<unsigned int>(syscall(SYS_gettid)), ##__VA_ARGS__); \
} while (0)

#define HCCL_DEBUG(format, ...) do { \
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG))) { \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_DEBUG, format, ##__VA_ARGS__); \
    } \
} while (0)

#define HCCL_INFO(format, ...) do { \
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) { \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_INFO, format, ##__VA_ARGS__); \
    } \
} while (0)

#define HCCL_WARNING(format, ...) do { \
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_WARN))) { \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_WARN, format, ##__VA_ARGS__); \
    } \
} while (0)

#define HCCL_ERROR(format, ...) do { \
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_ERROR))) { \
        HCCL_ERROR_LOG_PRINT(format, ##__VA_ARGS__); \
    } \
} while (0)

#define HCCL_RUN_INFO(format, ...) do { \
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO, HCCL_LOG_MASK))) { \
        HCCL_RUN_LOG_PRINT(format, ##__VA_ARGS__); \
    } \
} while (0)

#define HCCL_RUN_WARNING(format, ...) do { \
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_WARN, HCCL_LOG_MASK))) { \
        HCCL_LOG_PRINT(HCCL_LOG_MASK, HCCL_LOG_WARN, format, ##__VA_ARGS__); \
    } \
} while (0)

#define HCCL_ERROR_CODE(error) ((SYSTEM_RESERVE_ERROR << 32) + (HCCL_MODULE_ID << 24) + \
    ((static_cast<u64>(HcclSubModuleID::LOG_SUB_MODULE_ID_HCCL)) << 16) + static_cast<u64>(error))
#define HCOM_ERROR_CODE(error) ((SYSTEM_RESERVE_ERROR << 32) + (HCCL_MODULE_ID << 24) + \
    ((static_cast<u64>(HcclSubModuleID::LOG_SUB_MODULE_ID_HCOM)) << 16) + static_cast<u64>(error))

#define CHK_PTR_NULL(ptr) do { \
    if (UNLIKELY((ptr) == nullptr)) { \
        HCCL_ERROR("[%s]errNo[0x%016llx] ptr [%s] is nullptr, return HCCL_E_PTR", \
            __func__, HCCL_ERROR_CODE(HCCL_E_PTR), #ptr); \
        return HCCL_E_PTR; \
    } \
} while (0)

#define CHK_RET(call) do { \
    HcclResult hcclRet = (call); \
    if (UNLIKELY(hcclRet != HCCL_SUCCESS)) { \
        if (hcclRet == HCCL_E_AGAIN) { \
            HCCL_WARNING("[%s]call trace: hcclRet -> %d", __func__, static_cast<int>(hcclRet)); \
        } else { \
            HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, static_cast<int>(hcclRet)); \
        } \
        return hcclRet; \
    } \
} while (0)

#define HCCL_CHK_PTR(ptr) CHK_PTR_NULL(ptr)
#define HCCL_CHK_RET(call) CHK_RET(call)

#define ACLCHECK(call) do { \
    aclError aclRet = (call); \
    if (UNLIKELY(aclRet != ACL_SUCCESS)) { \
        HCCL_ERROR("[%s] acl call failed, ret=%d", __func__, static_cast<int>(aclRet)); \
        return HCCL_E_RUNTIME; \
    } \
} while (0)

#endif
