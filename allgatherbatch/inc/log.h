#ifndef HCCL_ALLGATHERBATCH_LOG_H
#define HCCL_ALLGATHERBATCH_LOG_H

#include <cstdio>
#include "hccl/hccl_types.h"

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_ERROR
#endif

enum LogLevel {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_NONE = 4,
};

#ifndef LIKELY
#define LIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 1)))
#define UNLIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 0)))
#endif

#ifdef HOST_COMPILE
#define HCCL_LOG(level, prefix, format, ...)                                \
    do {                                                                          \
        if (LOG_LEVEL <= level) {                                                 \
            std::printf("[%s][%s][%s:%d] " format "\n", prefix, __func__,       \
                __FILE__, __LINE__, ##__VA_ARGS__);                               \
        }                                                                         \
    } while (0)
#else
#define HCCL_LOG(level, prefix, format, ...) do {} while (0)
#endif

#define HCCL_DEBUG(format, ...) HCCL_LOG(LOG_LEVEL_DEBUG, "DEBUG", format, ##__VA_ARGS__)
#define HCCL_INFO(format, ...) HCCL_LOG(LOG_LEVEL_INFO, "INFO", format, ##__VA_ARGS__)
#define HCCL_WARNING(format, ...) HCCL_LOG(LOG_LEVEL_WARNING, "WARN", format, ##__VA_ARGS__)
#define HCCL_ERROR(format, ...) HCCL_LOG(LOG_LEVEL_ERROR, "ERROR", format, ##__VA_ARGS__)

#define HCCL_CHK_PTR(ptr)                                                    \
    do {                                                                           \
        if (UNLIKELY((ptr) == nullptr)) {                                          \
            HCCL_ERROR("pointer %s is null", #ptr);                         \
            return HCCL_E_PTR;                                                     \
        }                                                                          \
    } while (0)

#define HCCL_CHK_RET(call)                                                   \
    do {                                                                           \
        HcclResult hcclRet = (call);                                               \
        if (UNLIKELY(hcclRet != HCCL_SUCCESS)) {                                   \
            HCCL_ERROR("call failed, ret=%d", static_cast<int>(hcclRet));   \
            return hcclRet;                                                        \
        }                                                                          \
    } while (0)

#define ACLCHECK(call)                                                   \
    do {                                                                           \
        aclError aclRet = (call);                                                  \
        if (UNLIKELY(aclRet != ACL_SUCCESS)) {                                     \
            HCCL_ERROR("acl call failed, ret=%d", static_cast<int>(aclRet));\
            return HCCL_E_RUNTIME;                                                 \
        }                                                                          \
    } while (0)

#endif
