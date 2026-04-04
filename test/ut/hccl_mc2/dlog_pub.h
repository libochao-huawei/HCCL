#ifndef HCCL_MC2_UT_DLOG_PUB_H
#define HCCL_MC2_UT_DLOG_PUB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DLOG_DEBUG = 0,
    DLOG_INFO = 1,
    DLOG_WARN = 2,
    DLOG_ERROR = 3,
};

enum {
    HCCL = 0,
    RUN_LOG_MASK = 0,
};

void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif
