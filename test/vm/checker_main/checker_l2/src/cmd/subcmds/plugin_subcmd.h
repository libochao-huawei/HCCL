#ifndef HCCL_VM_PLUGIN_COMMAND_H
#define HCCL_VM_PLUGIN_COMMAND_H

#include <string>
#include "cmd_base.h"
#include "hccl_common_defs.h"

namespace HcclSim {
class PluginCommand : public CommandBase {
public:
    static std::string StaticName() { return "plugin"; }
    void Setup(CLI::App& app) override;
    
private:
    std::string plugName;
};
}

#endif