#include <string>
#include "log_subcmd.h"
#include "cmd_base_utils.h"
#include "hccl_vm_log.h"
#include "hccl_common_defs.h"

namespace HcclSim {
void LogCommand::Setup(CLI::App& app) {
    auto subCmdLog = app.add_subcommand("log", "日志管理");

    auto* levelOpt = subCmdLog->add_option("-l,--level", level_, "设置日志等级")
        ->transform(CLI::CheckedTransformer(levelMap_, CLI::ignore_case));
    auto* consoleLevelOpt = subCmdLog->add_option("--console-level", consoleLevel_, "设置console日志等级")
        ->transform(CLI::CheckedTransformer(levelMap_, CLI::ignore_case));
    auto* fileLevelOpt = subCmdLog->add_option("--file-level", fileLevel_, "设置file日志等级")
        ->transform(CLI::CheckedTransformer(levelMap_, CLI::ignore_case));

    levelOpt->excludes(consoleLevelOpt)->excludes(fileLevelOpt);
    consoleLevelOpt->excludes(levelOpt);
    fileLevelOpt->excludes(levelOpt);
    
    subCmdLog->callback([this]() { Execute(); });
}

void LogCommand::Execute() {
    if (consoleLevel_.has_value()) {
        HCCL_VM_DEBUG("input console-level: {}", consoleLevel_.value());
        SetConsoleLogLevel(consoleLevel_.value());
    }

    if (fileLevel_.has_value()) {
        HCCL_VM_DEBUG("input file-level: {}", fileLevel_.value());
        SetFileLogLevel(fileLevel_.value());
    }

    if (level_.has_value()) {
        HCCL_VM_DEBUG("input level: {}", level_.value());
        SetConsoleLogLevel(level_.value());
        SetFileLogLevel(level_.value());
    }

    ShowCurrentLogLevel();
}

static inline CommandAutoRegister<LogCommand> g_log_cmd_reg{};
}
