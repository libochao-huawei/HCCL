#include <string>
#include "start_subcmd.h"
#include "cmd_base_utils.h"
#include "hccl_vm_log.h"
#include "hccl_common_defs.h"

namespace HcclSim {
void StartCommand::Setup(CLI::App& app) {
    auto sub_start = app.add_subcommand("start", "start: 启动仿真环境,请勿在子bash中重复启用");

    sub_start->add_option("configFile", configFileName, "加载建模配置yaml文件")->required()->check(FileInModelDir);
    sub_start->add_option("--level", g_hcclVmLevel, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");
    
    sub_start->callback([this]() { Execute(); });
}

void StartCommand::Execute() {
    if (g_hcclVmBashFlag) {
        HCCL_VM_WARN("[HVM] hccl-vm 已经启动, 请勿在子bash中再次启动");
        return;
    }
    HCCL_VM_INFO("[HVM] Initializing: Model={}, Level={}", configFileName, g_hcclVmLevel);
    std::vector<std::string> serverIdx2Ip;
    if (!ParseYamlTopo(configFileName, topoMeta, serverIdx2Ip)) {
        return;
    }
    auto ret = InitHvmEnv(topoMeta, serverIdx2Ip , g_hcclVmLevel);
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("[HVM] 初始化模拟环境失败，正在清理环境");
        auto cleanRet = HcclVmExit();
        if (cleanRet != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
            HCCL_VM_ERROR("[HVM] 清理环境失败，请检查环境残留");
        }
        return;
    }
    StartHostServer();
    return;
}

static inline CommandAutoRegister<StartCommand> g_start_cmd_reg{};
}
