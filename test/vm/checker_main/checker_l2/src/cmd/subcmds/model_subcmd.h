#ifndef HCCL_VM_MODEL_COMMAND_H
#define HCCL_VM_MODEL_COMMAND_H

#include <string>
#include "cmd_base.h"
#include "hccl_common_defs.h"

namespace HcclSim {
class ModelCommand : public CommandBase {
public:
    static std::string StaticName() { return "model"; }
    void Setup(CLI::App& app) override;
    
private:
    void Execute();
};
}

#endif