#ifndef HCCL_VM_LOG_COMMAND_H
#define HCCL_VM_LOG_COMMAND_H

#include <string>
#include <optional>
#include <map>
#include "cmd_base.h"
#include "hccl_common_defs.h"

namespace HcclSim {
class LogCommand : public CommandBase {
public:
    static std::string StaticName() { return "log"; }
    void Setup(CLI::App& app) override;
    
private:
    void Execute();
    
    std::optional<int> level_;
    std::optional<int> consoleLevel_;
    std::optional<int> fileLevel_;

    const std::map<std::string, int> levelMap_ {
            {"trace", 0},
            {"debug", 1},
            {"info", 2},
            {"warn", 3},
            {"warning", 3},
            {"error", 4},
            {"err", 4},
            {"critical", 5},
    };
};
}

#endif