#ifndef OP_LOG_H_
#define OP_LOG_H_
#include <iostream>
#include <string>
#include <cstdarg>
#include <stdlib.h>

#ifdef STUB_LOG
#define OP_LOGD(...)
#define OP_LOGI(...)
#define OP_LOGE(...)
#define OP_LOGW(...)
#define OP_EVENT(...)
#else
static const char *g_open_log = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
static const char *g_log_level = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
static const char *g_event_enable = std::getenv("ASCEND_GLOBAL_EVENT_ENABLE");
inline void OP_LOGD(const std::string &op_name, const std::string &log, ...) {
  if (g_open_log == nullptr) {
    return;
  }
  if (g_log_level != nullptr) {
    if (std::atoi(g_log_level) <= 0) {
      std::string slog = "[DEBUG][" + op_name + "]" + log + "\n";
      va_list args;
      va_start(args, log);
      vprintf(slog.c_str(), args);
      va_end(args);
    }
  }
}


inline void OP_LOGI(const std::string &op_name, const std::string &log, ...) {
  if (g_open_log == nullptr) {
    return;
  }
  if (g_log_level != nullptr) {
    if (std::atoi(g_log_level) <= 1) {
      std::string slog = "[INFO][" + op_name + "]" + log + "\n";
      va_list args;
      va_start(args, log);
      vprintf(slog.c_str(), args);
      va_end(args);
    }
  }
}


inline void OP_LOGW(const std::string &op_name, const std::string &log, ...) {
  if (g_open_log == nullptr) {
    return;
  }
  if (g_log_level != nullptr) {
    if (std::atoi(g_log_level) <= 2) {
      std::string slog = "[WARNING][" + op_name + "]" + log + "\n";
      va_list args;
      va_start(args, log);
      vprintf(slog.c_str(), args);
      va_end(args);
    }
  }
}


inline void OP_LOGE(const std::string &op_name, const std::string &log, ...) {
  if (g_open_log == nullptr) {
    return;
  }
  if (g_log_level != nullptr) {
    if (std::atoi(g_log_level) <= 3) {
      std::string slog = "[ERROR][" + op_name + "]" + log + "\n";
      va_list args;
      va_start(args, log);
      vprintf(slog.c_str(), args);
      va_end(args);
    }
  }
}

inline void OP_EVENT(const std::string &op_name, const std::string &log, ...) {
  if (g_open_log == nullptr) {
    return;
  }
  if (g_event_enable != nullptr) {
    if (std::atoi(g_event_enable) == 1) {
      std::string slog = "[EVENT][" + op_name + "]" + log + "\n";
      va_list args;
      va_start(args, log);
      vprintf(slog.c_str(), args);
      va_end(args);
    }
  }
}
#endif
#endif