#ifndef HCCL_VM_START_COMMAND_H
#define HCCL_VM_START_COMMAND_H

#include <string>
#include "cmd_base.h"
#include "hccl_common_defs.h"

namespace HcclSim {
class StartCommand : public CommandBase {
public:
    static std::string StaticName() { return "start"; }
    void Setup(CLI::App& app) override;
    
private:
    void Execute();
    
    std::string configFileName;
    TopoMeta topoMeta;
};
}

#endif