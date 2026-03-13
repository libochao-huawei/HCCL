#ifndef SEMANTICS_UTILS_H
#define SEMANTICS_UTILS_H

#include "log.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"

namespace checker {

std::string GenMsg(const char* fileName, s32 lineNum, const char *fmt, ...);

#define DUMP_AND_ERROR(...)                                                               \
    do {                                                                                  \
        HCCL_ERROR(__VA_ARGS__);                                                          \
        std::string msg = GenMsg(__FILE__, __LINE__, __VA_ARGS__);                        \
        DataDumper::Global()->AddErrorString(msg);                                        \
    } while (0)

#define CHECKER_WARNING_LOG(format, ...) do {                                             \
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_WARN, INVLID_MOUDLE_ID))) {                   \
        HCCL_LOG_PRINT(INVLID_MOUDLE_ID, HCCL_LOG_WARN, format, ##__VA_ARGS__);           \
    }                                                                                     \
} while(0)

#define CHECKER_WARNING(...)                                                              \
    do {                                                                                  \
        CHECKER_WARNING_LOG(__VA_ARGS__);                                                 \
        std::string msg = GenMsg(__FILE__, __LINE__, __VA_ARGS__);                        \
        DataDumper::Global()->AddErrorString(msg);                                        \
    } while (0)
}

#endif