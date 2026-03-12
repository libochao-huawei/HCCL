#include <string>
#include "plugin_subcmd.h"
#include "cmd_base_utils.h"
#include "hccl_vm_log.h"
#include "hccl_common_defs.h"

namespace HcclSim {
void PluginCommand::Setup(CLI::App& app) {
    auto sub_plugin = app.add_subcommand("plugin", "插件管理子命令");
    sub_plugin->require_subcommand(1);
    // install
    auto plugin_install = sub_plugin->add_subcommand("install", "安装插件");
    plugin_install->add_option("name", plugName, "插件文件名")->required();
    plugin_install->callback([&]() {
        auto ret = InstallUserPlugin(plugName);
    });
    // uninstall
    auto plugin_uninstall = sub_plugin->add_subcommand("uninstall", "卸载插件");
    plugin_uninstall->add_option("name", plugName, "插件文件名")->required()
        ->check([](const std::string &value) -> std::string {
            if (value.length() > 1 && value[0] == '@') {
                return ""; // 返回空串表示通过
            }
            return "[HVM] [ERROR] Uninstall plugin : Invalid format! Plugin name must start with '@' (e.g., @myplugin).";
        });
    plugin_uninstall->callback([&]() {
        auto ret = UninstallUserPlugin(plugName);
    });
    // run
    auto plugin_run = sub_plugin->add_subcommand("run", "运行插件"); // todo
    plugin_run->add_option("name", plugName, "插件文件名")->required()
        ->check([](const std::string &value) -> std::string {
            if (value.length() > 1 && value[0] == '@') {
                return ""; // 返回空串表示通过
            }
            return "[HVM] [ERROR] Run plugin : Invalid format! Plugin name must start with '@' (e.g., @myplugin).";
        });
    plugin_run->callback([&]() {
        auto ret = RunUserPlugin(plugName);
    });
    // list
    auto plugin_list = sub_plugin->add_subcommand("list", "展示已安装插件");
    plugin_list->callback([&]() {
        ShowUserPlugin();
    });
}

static inline CommandAutoRegister<PluginCommand> g_plugin_cmd_reg{};
}
