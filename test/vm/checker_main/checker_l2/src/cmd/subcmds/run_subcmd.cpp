#include <string>
#include "run_subcmd.h"
#include "cmd_base_utils.h"
#include "hccl_vm_log.h"
#include "hccl_common_defs.h"

namespace HcclSim {
void RunCommand::Setup(CLI::App& app) {
    auto sub_oneShot = app.add_subcommand("run", "算例one_shot运行模式, 请勿在子bash中重复启用,命令后必须接有算例执行指令");

    sub_oneShot->add_option("configFile", configFileName, "加载建模配置yaml文件")->required()->check(FileInModelDir);
    sub_oneShot->add_option("--level", g_hcclVmLevel, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");
    sub_oneShot->allow_extras(true);
    
    sub_oneShot->callback([this, &app]() { Execute(app); });
}

void RunCommand::Execute(CLI::App& app) {
    if (g_hcclVmBashFlag) {
        HCCL_VM_WARN("[HVM] hccl-vm 已经启动, 请勿在子bash中one_shot运行算例,请退出子bash再尝试");
        return;
    }
    CLI::App* tmp_cmd = app.get_subcommand("run");
    std::vector<std::string> leftargvs = tmp_cmd->remaining();
    if (leftargvs.empty()) {
        HCCL_VM_ERROR("[HVM] one_shot模式, 必须接有算例运行指令");
        return;
    }
    HCCL_VM_INFO("[HVM] Initializing: Model={}, Level={}", configFileName, g_hcclVmLevel);
    std::vector<std::string> serverIdx2Ip;
    if (!ParseYamlTopo(configFileName, topoMeta, serverIdx2Ip)) return;
    auto ret = InitHvmEnv(topoMeta, serverIdx2Ip, g_hcclVmLevel);
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("[HVM] 初始化模拟环境失败，正在清理环境");
        auto cleanRet = HcclVmExit();
        if (cleanRet != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
            HCCL_VM_ERROR("[HVM] 清理环境失败，请检查环境残留");
        }
        return;
    }
    CstyleCmd syscmd(leftargvs);
    HCCL_VM_INFO("[HVM] one_shot模式, 执行: {}", syscmd.cmd());
    std::string proxyPath = GetBinLocation() + "/libhccl_proxy_level2.so";
    setenv("LD_PRELOAD", proxyPath.c_str(), 1);
    int sysRet = std::system(syscmd.cmd().c_str()); // system() 会阻塞当前进程直到子命令结束
    if (sysRet != 0) {
        HCCL_VM_ERROR("[HVM] 算例执行失败: {}", sysRet);
    }
    RemoveFromLDPreload(proxyPath);
    auto cleanRet = HcclVmExit();
    return;
}

static inline CommandAutoRegister<RunCommand> g_run_cmd_reg{};
}
