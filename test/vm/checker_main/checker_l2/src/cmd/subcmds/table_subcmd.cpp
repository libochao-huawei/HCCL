#include <string>
#include "table_subcmd.h"
#include "cmd_table_utils.h"
#include "hccl_vm_log.h"
#include "hccl_common_defs.h"
#include "sim_runner_db.h"

namespace HcclSim {
void TableCommand::Setup(CLI::App& app) {
    auto tableCmd = app.add_subcommand("table", "table opt");
    tableCmd->require_subcommand(1);
    auto showCmd = tableCmd->add_subcommand("show", "show table");

    showCmd->add_option("name", showStr, "show table all")->required();
    showCmd->callback([this]() {
        if (showStr == "all") {
            std::vector<std::string> tables;
            tables = RunnerDB::GetAllTableName();
            for (auto &tbl : tables) {
                std::cout << tbl << std::endl;
            }
        } else {
            CmdTableShow(showStr);
        }
    });

    auto updateCmd = tableCmd->add_subcommand("update", "update table");

    updateCmd->add_option("table", tableName, "Table name (e.g., Device, Server, Host)")->required();
    updateCmd->add_option("id", rowId, "Row ID to update")->required();
    updateCmd->add_option("column", columnName, "Column to update (e.g., soc_version)")->required();
    updateCmd->add_option("value", newValue, "New value (as string)")->required();

    updateCmd->callback([&]() { CmdTableUpdate(tableName, rowId, columnName, newValue); });
}

static inline CommandAutoRegister<TableCommand> g_table_cmd_reg{};
}
