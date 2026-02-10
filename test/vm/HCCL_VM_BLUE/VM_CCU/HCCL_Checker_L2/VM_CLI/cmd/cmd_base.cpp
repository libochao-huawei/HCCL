/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include "hccl_common_defs.h"
#include "cmd_utils.h"
#include "cmd_base.h"

namespace {
    class Start : public BaseCommand {
    public:
        void setup(CLI::App& app) override {
            auto sub_start = app.add_subcommand("start", "start: 启动仿真环境,请勿在子bash中重复启用");

            sub_start->add_option("configFile", configFileName, "加载建模配置yaml文件")->required()->check(FileInModelDir);
            sub_start->add_option("--level", g_hcclVmLevel, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");

            sub_start->callback([&]() {
                if (g_hcclVmBashFlag) {
                    HCCL_VM_WARN("[HVM] hccl-vm 已经启动, 请勿在子bash中再次启动");
                    return;
                }
                HCCL_VM_INFO("[HVM] Initializing: Model={}, Level={}", configFileName, g_hcclVmLevel);
                if (!ParseYamlTopo(configFileName, topoMeta)) return;
                auto ret = InitHvmEnv(topoMeta, g_hcclVmLevel);
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
            });
        }
    private:
        std::string configFileName;
        TopoMeta topoMeta;
    };

    class Run : public BaseCommand {
    public:
        void setup(CLI::App& app) override {
            auto sub_oneShot = app.add_subcommand("run", "算例one_shot运行模式, 请勿在子bash中重复启用,命令后必须接有算例执行指令");
            sub_oneShot->add_option("configFile", configFileName, "加载建模配置yaml文件")->required()->check(FileInModelDir);
            sub_oneShot->add_option("--level", g_hcclVmLevel, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");
            sub_oneShot->allow_extras(true);
            sub_oneShot->callback([&]() {
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
                if (!ParseYamlTopo(configFileName, topoMeta)) return;
                auto ret = InitHvmEnv(topoMeta, g_hcclVmLevel);
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
            });
        }
    private:
        std::string configFileName;
        TopoMeta topoMeta;
        std::vector<std::string> leftargvs;
    };

    class Model : public BaseCommand {
    public:
        void setup(CLI::App& app) override {
            auto sub_model = app.add_subcommand("model", "管理建模文件");
            sub_model->require_subcommand(1);
            auto model_list = sub_model->add_subcommand("list", "展示建模文件");
            model_list->callback([&]() {
                ShowModel();
            });
        }
    };

    class Plugin : public BaseCommand {
    public:
        void setup(CLI::App& app) override {
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
                auto ret = ShowUserPlugin();
            });
        }
    private:
        std::string plugName;
    };
}

void RegisterCommands(std::vector<std::unique_ptr<BaseCommand>>& cmds) {
    cmds.push_back(std::make_unique<Start>());
    cmds.push_back(std::make_unique<Run>());
    cmds.push_back(std::make_unique<Model>());
    cmds.push_back(std::make_unique<Plugin>());
}