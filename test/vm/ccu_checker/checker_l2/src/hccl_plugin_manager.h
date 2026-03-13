#ifndef HCCL_PLUGIN_MANAGER_H
#define HCCL_PLUGIN_MANAGER_H

#include "hccl_common_defs.h"
#include "hccl_plugin.h"
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <string>

class HcclPluginManager {
public:
    static HcclPluginManager& GetInstance();
    // 注册插件 同时通过LoadPlugins加载
    HcclSim::HcclVmResult RegisterPlugin(const std::string& pluginTag);

    HcclSim::HcclVmResult SendMessageToPlugin(const std::string& pluginTag, const std::string& action, const nlohmann::json& payload = nlohmann::json::object());
    // 广播给所有运行中的插件
    HcclSim::HcclVmResult BroadcastToAllPlugin(const std::string& action, const nlohmann::json& payload = nlohmann::json::object());

    // 获取已加载的插件状态
    std::vector<std::string> GetPluginStatus() const;

    // 运行插件
    std::vector<HcclSim::HcclVmResult> StartPlugins(const std::vector<std::string>& tag);
    // 关闭并卸载插件
    std::vector<HcclSim::HcclVmResult> StopPlugins(const std::vector<std::string>& tag);

    HcclSim::HcclVmResult StopAllPlugins();

    // 禁止拷贝
    HcclPluginManager(const HcclPluginManager&) = delete;
    HcclPluginManager& operator=(const HcclPluginManager&) = delete;

private:
    HcclPluginManager() = default;
    ~HcclPluginManager() { StopAllPlugins(); };

    bool GetPluginFolderPath(const std::string& pluginTag, std::string& pluginFolderPath);
    bool IsMatchingPlugin(const std::string& manifestPath, const std::string& targetTag);
    std::map<std::string, std::shared_ptr<HcclPlugin>> m_plugins; // 插件Tag - 插件

    mutable std::mutex m_mutex; // 线程安全锁
};

#endif // HCCL_PLUGIN_MANAGER_H