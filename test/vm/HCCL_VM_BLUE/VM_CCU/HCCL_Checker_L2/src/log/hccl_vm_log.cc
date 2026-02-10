#include <fstream>
#include "hccl_vm_log.h"
#include "spdlog/details/os-inl.h"
#include "zlib.h"

spdlog::logger* g_logger{nullptr};

std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> InitConsoleSink(const LogConfig& config)
{
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_level(static_cast<spdlog::level::level_enum>(config.consoleLevel));
    sink->set_pattern("[%^%l%$][PID:%P][TID:%t][%s][%!] %v");
    return sink;
}

bool CompressFileGz(const std::string& source, const std::string& dest) {
    std::ifstream inFile(source, std::ios::binary);
    if (!inFile) {
        std::cout << "[ERROR] Cannot open log file: " << source << std::endl;
        return false;
    }

    gzFile outFile = gzopen(dest.c_str(), "wb");
    if (!outFile) {
        std::cout << "[ERROR] Cannot create compressed file: " << dest << std::endl;
        inFile.close();
        return false;
    }

    gzsetparams(outFile, Z_DEFAULT_COMPRESSION, Z_DEFAULT_STRATEGY);    // compress config

    constexpr size_t bufferSize = 4 * 1024; // 4KB缓冲区
    char buffer[bufferSize];

    // 读取、压缩并写入
    while (inFile.read(buffer, sizeof(buffer)) || inFile.gcount()) {
        int bytesRead = inFile.gcount();
        int bytesWritten = gzwrite(outFile, buffer, bytesRead);

        if (bytesWritten != bytesRead) {
            std::cout << "[ERROR] Failed to compress: " << bytesRead << " -> " << bytesWritten << std::endl;
            inFile.close();
            gzclose(outFile);
            return false;
        }
    }

    inFile.close();
    gzclose(outFile);

    return true;
}

std::shared_ptr<spdlog::sinks::rotating_file_sink_mt> InitFileSink(const LogConfig& config)
{
    // after close handler
    spdlog::file_event_handlers handlers;
    handlers.after_close = [config](const std::string& filePath) {
        // rename
        static std::atomic<uint32_t> g_log_file_index{0};
        if (filePath.size() < config.fileSuffix.size() ||
            filePath.substr(filePath.size() - config.fileSuffix.size()) != config.fileSuffix) {
            std::cout << "[ERROR] Log file name error!" << std::endl;
            return;
        }
        std::ostringstream oss;
        oss << filePath.substr(0, filePath.size() - config.fileSuffix.size())
            << "_" << std::to_string(g_log_file_index.fetch_add(1, std::memory_order_relaxed))
            << config.fileSuffix;
        const std::string newPath = oss.str();
        if (spdlog::details::os::rename(filePath, newPath) != 0) {
            std::cout << "[ERROR] Fail to rename rotating log file: " << newPath << std::endl;
            return;
        }

        // compress
        if (config.enableCompress) {
            std::string gzPath = newPath + ".gz";
            if (!CompressFileGz(newPath, gzPath)) {
                std::cout << "[ERROR] Fail to compress rotating log file: " << newPath << std::endl;
                return;
            }
            // delete old
            if (spdlog::details::os::remove_if_exists(newPath) != 0) {
                std::cout << "[ERROR] Fail to remove!" << std::endl;
                return;
            }
        }
    };

    // log file name
    std::ostringstream logFileName;
    logFileName << config.filePath << "/" << config.fileBaseName << "_" << std::to_string(getpid()) << config.fileSuffix;

    auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        logFileName.str(), config.maxFileSize, config.maxFiles, true, handlers);
    sink->set_level(static_cast<spdlog::level::level_enum>(config.fileLevel));
    sink->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [%l] [PID:%P] [TID:%t] [%s:%#] %v");
    return sink;
}

void InitLogger(const LogConfig& config)
{
    try {
        auto console_sink = InitConsoleSink(config);
        auto file_sink = InitFileSink(config);
        g_logger = new spdlog::logger("muti-logger", spdlog::sinks_init_list({console_sink, file_sink}));
        g_logger->set_level(spdlog::level::trace);  // global级别设置为最低, 只通过sink的级别控制日志输出
        g_logger->flush_on(spdlog::level::warn);

        std::atexit([] (){ DeInitLogger(); });
    } catch (...) {
        std::cout << "[ERROR] Logger init failed" << std::endl;
    }
}

void DeInitLogger()
{
    if (g_logger != nullptr) {
        delete g_logger;
        g_logger = nullptr;
    }
}
