#include <string>
#include "model_subcmd.h"
#include "cmd_base_utils.h"
#include "hccl_vm_log.h"
#include "hccl_common_defs.h"

namespace HcclSim {
void ModelCommand::Setup(CLI::App& app) {
    auto sub_model = app.add_subcommand("model", "管理建模文件");

    sub_model->require_subcommand(1);
    auto model_list = sub_model->add_subcommand("list", "展示建模文件");
    
    model_list->callback([this]() { Execute(); });
}

void ModelCommand::Execute() {
    ShowModel();
}

static inline CommandAutoRegister<ModelCommand> g_model_cmd_reg{};
}
