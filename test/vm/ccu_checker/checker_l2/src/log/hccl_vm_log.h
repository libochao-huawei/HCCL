#ifndef HCCL_VM_LOG_H
#define HCCL_VM_LOG_H

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE // global级别设置为最低, 只通过sink的级别控制日志输出

#include <iostream>
#include <sstream>
#include <memory>
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

extern spdlog::logger* g_logger;

#define HCCL_VM_TRACE(...)                                  \
    do {                                                    \
        if (g_logger != nullptr) {                          \
            SPDLOG_LOGGER_TRACE(g_logger, __VA_ARGS__);     \
        }                                                   \
    } while(0)

#define HCCL_VM_DEBUG(...)                                  \
    do {                                                    \
        if (g_logger != nullptr) {                          \
            SPDLOG_LOGGER_DEBUG(g_logger, __VA_ARGS__);     \
        }                                                   \
    } while(0)

#define HCCL_VM_INFO(...)                                   \
    do {                                                    \
        if (g_logger != nullptr) {                          \
            SPDLOG_LOGGER_INFO(g_logger, __VA_ARGS__);      \
        }                                                   \
    } while(0)

#define HCCL_VM_WARN(...)                                   \
    do {                                                    \
        if (g_logger != nullptr) {                          \
            SPDLOG_LOGGER_WARN(g_logger, __VA_ARGS__);      \
        }                                                   \
    } while(0)

#define HCCL_VM_ERROR(...)                                  \
    do {                                                    \
        if (g_logger != nullptr) {                          \
            SPDLOG_LOGGER_ERROR(g_logger, __VA_ARGS__);     \
        }                                                   \
    } while(0)

#define HCCL_VM_CRITICAL(...)                               \
    do {                                                    \
        if (g_logger != nullptr) {                          \
            SPDLOG_LOGGER_CRITICAL(g_logger, __VA_ARGS__);  \
        }                                                   \
    } while(0)


struct LogConfig
{
    int consoleLevel{3};                // default warn
    int fileLevel{1};                   // default debug
    size_t maxFileSize{10*1024*1024};   // 10MB
    size_t maxFiles{UINT16_MAX};
    std::string filePath{"logs"};
    std::string fileBaseName{"app_log"};
    std::string fileSuffix{".log"};
    bool enableCompress{false};
};

void InitLogger(const LogConfig& config);
void DeInitLogger();
std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> InitConsoleSink(const LogConfig& config);
std::shared_ptr<spdlog::sinks::rotating_file_sink_mt> InitFileSink(const LogConfig& config);

#endif //HCCL_VM_LOG_H
