#ifndef HCCL_VM_RUN_COMMAND_H
#define HCCL_VM_RUN_COMMAND_H

#include <string>
#include "cmd_base.h"
#include "hccl_common_defs.h"

namespace HcclSim {
class RunCommand : public CommandBase {
public:
    static std::string StaticName() { return "run"; }
    void Setup(CLI::App& app) override;
    
private:
    void Execute(CLI::App& app);
    
    std::string configFileName;
    TopoMeta topoMeta;
    std::vector<std::string> leftargvs;
};
}

#endif