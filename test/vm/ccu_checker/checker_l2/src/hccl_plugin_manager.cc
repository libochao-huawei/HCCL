// host/src/hccl_plugin_manager.cpp
#include "hccl_plugin_manager.h"
#include "hccl_vm_subcmd.h"
#include <mutex>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <string>
#include <stdexcept>
#include <cstring>
#include <libgen.h>
#include <stack>
#include <dirent.h>
#include "hccl_vm_log.h"

using namespace HcclSim;

// 插件管理器实现（单例，作为Host的内部组件）
HcclPluginManager& HcclPluginManager::GetInstance()
{
    static HcclPluginManager instance;
    return instance;
}

bool HcclPluginManager::IsMatchingPlugin(const std::string& manifestPath, const std::string& targetTag)
{
    // 1. 预检查：文件是否可读
    if (access(manifestPath.c_str(), R_OK) != 0) {
        return false;
    }

    try {
        std::ifstream f(manifestPath);
        if (!f.is_open()) {
            return false;
        }

        // 2. 解析 JSON
        nlohmann::json data = nlohmann::json::parse(f);

        // 3. 字段校验与匹配
        // 使用我们定义的静态常量 Manifest::pluginName
        const std::string& nameKey = HcclPlugin::Manifest::pluginName;

        if (data.contains(nameKey) && data[nameKey].is_string()) {
            std::string currentPluginName = data[nameKey].get<std::string>();
            
            // 进行 Tag 匹配
            if (currentPluginName == targetTag) {
                // 如果必需字段缺失，认为是一个损坏的插件，返回 false 并记录日志
                if (!data.contains(HcclPlugin::Manifest::pluginEntry)) {
                    HCCL_VM_ERROR("[Plugin] [{}] '{}' missing entry field.", __func__, targetTag);
                    return false;
                }
                return true;
            }
        }
    } catch (const nlohmann::json::parse_error& e) {
        // 生产环境建议记录具体的解析错误位置
        HCCL_VM_ERROR("[Plugin] [{}] JSON Parse Error at byte {}", __func__, e.byte);
        return false;
    } catch (const std::exception& e) {
        // 处理其他可能的异常（如文件读取中途断开等）
        HCCL_VM_ERROR("[Plugin] [{}] Error: ", __func__, e.what());
        return false;
    }

    return false;
}

bool HcclPluginManager::GetPluginFolderPath(const std::string& pluginTag, std::string& pluginFolderPath)
{
    std::string rootDir = GetBinLocation() + HcclPlugin::PLUGIN_PATH;
    std::string foundPath = "";
    bool hasConflict = false;

    struct ScanTask {
        std::string path;
        int depth;
    };

    std::stack<ScanTask> taskStack;
    taskStack.push({rootDir, 0});

    while (!taskStack.empty()) {
        ScanTask current = taskStack.top();
        taskStack.pop();

        // 1. 检查当前目录是否包含匹配的 manifest
        std::string manifestPath = current.path + HcclPlugin::MANIFEST_FILE;
        if (IsMatchingPlugin(manifestPath, pluginTag)) {
            char absPath[PATH_MAX];
            if (realpath(current.path.c_str(), absPath)) {
                std::string currentAbsPath(absPath);
                
                // 冲突检查：如果之前已经找到了一个路径，且不是当前路径
                if (!foundPath.empty() && foundPath != currentAbsPath) {
                    HCCL_VM_ERROR("[Plugin] [{}] Conflict detected: Tag '{}' exists in both: {} and {}",
                        __func__, pluginTag, foundPath, currentAbsPath);
                    hasConflict = true;
                    break; 
                }
                foundPath = currentAbsPath;
            }
            // 按照约定，插件文件夹内部不再嵌套插件，找到后不进入子目录
            continue;
        }

        // 2. 限制深度进行 DFS 遍历
        if (current.depth < HcclPlugin::MAX_SCAN_DEPTH) {
            DIR* dir = opendir(current.path.c_str());
            if (!dir) continue;

            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string name = entry->d_name;
                if (name == "." || name == "..") continue;

                std::string nextPath = current.path + "/" + name;
                struct stat st;
                // 使用 lstat 避免符号链接环路
                if (lstat(nextPath.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                    taskStack.push({nextPath, current.depth + 1});
                }
            }
            closedir(dir);
        }
    }

    // 3. 最终结果逻辑
    if (hasConflict) {
        pluginFolderPath = "";
        return false;
    }

    if (!foundPath.empty()) {
        pluginFolderPath = foundPath;
        return true;
    }

    return false;
}

std::vector<std::string> HcclPluginManager::GetPluginStatus() const {
    std::vector<std::string> result;
    
    // 1. 定义表头 (使用左对齐和固定宽度确保整齐)
    std::stringstream header;
    header << std::left << std::setw(20) << "PLUGIN_TAG" 
           << std::setw(10) << "PID" 
           << std::setw(15) << "STATUS";
    result.push_back(header.str());

    // 2. 遍历插件获取数据
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& pair : m_plugins) {
        const std::string& tag = pair.first;
        const auto& plugin = pair.second;

        // 获取 PID 和 运行状态
        int32_t pid = plugin->GetPid(); // 确保 HcclPlugin 有 GetPid() 方法
        bool running = plugin->IsRunning();
        std::string statusStr = running ? "RUNNING" : "STOPPED/EXITED";

        std::stringstream row;
        row << std::left << std::setw(20) << tag 
            << std::setw(10) << (pid > 0 ? std::to_string(pid) : "N/A") 
            << std::setw(15) << statusStr;
        
        result.push_back(row.str());
    }

    return result;
}

HcclVmResult HcclPluginManager::RegisterPlugin(const std::string& pluginTag)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // 防止重复注册
    if (m_plugins.find(pluginTag) != m_plugins.end()) {
        HCCL_VM_WARN("[PluginManager] [{}] Plugin {} already registered", __func__, pluginTag);
        return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;
    }

    std::string folderPath;
    if (GetPluginFolderPath(pluginTag, folderPath) == false) {
        return HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND;
    }

    try {
        // 创建即启动 (符合你 HcclPlugin 构造函数的逻辑)
        auto plugin = std::make_shared<HcclPlugin>(folderPath);
        m_plugins[pluginTag] = plugin;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("[PluginManager] [{}] Failed to register plugin {}: {}", __func__, pluginTag, e.what());
        return HcclSim::HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclPluginManager::SendMessageToPlugin(const std::string& pluginTag, const std::string& action, const nlohmann::json& payload)
{
    auto it = m_plugins.find(pluginTag);
    if (it == m_plugins.end()) {
        return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    HcclVmResult ret = it->second->SendMessage(PLUGIN_MESSAGE_TYPE::MESSAGE, action, payload);
    return ret;
}

HcclVmResult HcclPluginManager::BroadcastToAllPlugin(const std::string& action, const nlohmann::json& payload)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    HcclVmResult ret = HcclVmResult::HCCL_SIM_SUCCESS;
    for (auto& pair : m_plugins) {
        if (pair.second->IsRunning()) {
            HcclVmResult singleRet = pair.second->SendMessage(PLUGIN_MESSAGE_TYPE::BROADCAST, action, payload);
            if (singleRet != HcclVmResult::HCCL_SIM_SUCCESS) {
                ret = HcclVmResult::HCCL_SIM_E_INTERNAL;
            }
        }
    }
    return ret;
}

std::vector<HcclSim::HcclVmResult> HcclPluginManager::StartPlugins(const std::vector<std::string>& tags) {
    std::vector<HcclSim::HcclVmResult> results;
    std::lock_guard<std::mutex> lock(m_mutex);

    for (const auto& tag : tags) {
        auto it = m_plugins.find(tag);
        if (it == m_plugins.end()) {
            results.push_back(HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
            continue;
        }

        // 如果插件已经在运行，通常视为成功或检查状态
        if (it->second->IsRunning()) {
            results.push_back(HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
        } else {
            // 调用 HcclPlugin 的 Start 方法
            results.push_back(it->second->Start());
        }
    }
    return results;
}

std::vector<HcclSim::HcclVmResult> HcclPluginManager::StopPlugins(const std::vector<std::string>& tags) {
    std::vector<HcclSim::HcclVmResult> results;
    std::lock_guard<std::mutex> lock(m_mutex);

    for (const auto& tag : tags) {
        auto it = m_plugins.find(tag);
        if (it == m_plugins.end()) {
            results.push_back(HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
            continue;
        }

        // 1. 调用插件内部的优雅停止逻辑（含5秒超时）
        HcclSim::HcclVmResult res = it->second->Stop();
        results.push_back(res);

        // 2. 从管理 Map 中移除（卸载）
        // 即使 Stop 失败（如超时），我们通常也从 Manager 中移除它，将其交由系统清理或用户手动处理
        m_plugins.erase(it);
    }
    return results;
}

HcclSim::HcclVmResult HcclPluginManager::StopAllPlugins() {
    // 1. 获取当前所有插件的 Tag
    std::vector<std::string> allTags;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& pair : m_plugins) {
            allTags.push_back(pair.first);
        }
    }

    // 2. 调用批量停止方法
    if (allTags.empty()) return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;

    StopPlugins(allTags);

    return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;
}