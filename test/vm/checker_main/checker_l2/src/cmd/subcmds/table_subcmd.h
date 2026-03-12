#ifndef HCCL_VM_START_COMMAND_H
#define HCCL_VM_START_COMMAND_H

#include <string>
#include "cmd_base.h"
#include "hccl_common_defs.h"

namespace HcclSim {
class TableCommand : public CommandBase {
public:
    static std::string StaticName() { return "table"; }
    void Setup(CLI::App& app) override;

    std::string showStr;
    std::string tableName;
    std::string columnName;
    uint64_t rowId = 0;
    std::string newValue;
};
}

#endif