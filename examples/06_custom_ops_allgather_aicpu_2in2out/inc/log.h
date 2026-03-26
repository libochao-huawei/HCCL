#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_LOG_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_LOG_H

#include <cstdio>
#include "hccl/hccl_types.h"
#include "acl/acl.h"

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
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
#define HCCL_DEBUG(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_DEBUG) { \
    std::printf("[DEBUG][%s][%s:%d] " format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
} } while (0)
#define HCCL_INFO(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_INFO) { \
    std::printf("[INFO][%s][%s:%d] " format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
} } while (0)
#define HCCL_WARNING(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_WARNING) { \
    std::printf("[WARN][%s][%s:%d] " format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
} } while (0)
#define HCCL_ERROR(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_ERROR) { \
    std::printf("[ERROR][%s][%s:%d] " format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
} } while (0)
#else
#define HCCL_DEBUG(format, ...) do {} while (0)
#define HCCL_INFO(format, ...) do {} while (0)
#define HCCL_WARNING(format, ...) do {} while (0)
#define HCCL_ERROR(format, ...) do {} while (0)
#endif

#define CHK_PTR_NULL(ptr) do { \
    if (UNLIKELY((ptr) == nullptr)) { \
        HCCL_ERROR("ptr [%s] is nullptr", #ptr); \
        return HCCL_E_PTR; \
    } \
} while (0)

#define CHK_PRT_RET(result, exeLog, retCode) do { \
    if (UNLIKELY(result)) { \
        exeLog; \
        return retCode; \
    } \
} while (0)

#define CHK_RET(call) do { \
    int32_t hcclRet = call; \
    if (UNLIKELY(hcclRet != HCCL_SUCCESS)) { \
        HCCL_ERROR("call failed, ret=%d", hcclRet); \
        return static_cast<HcclResult>(hcclRet); \
    } \
} while (0)

#define ACLCHECK(cmd) do { \
    aclError aclRet = cmd; \
    if (UNLIKELY(aclRet != ACL_SUCCESS)) { \
        HCCL_ERROR("acl call failed, ret=%d", aclRet); \
        return HCCL_E_RUNTIME; \
    } \
} while (0)

#endif
