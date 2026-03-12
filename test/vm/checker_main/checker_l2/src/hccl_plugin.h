#ifndef HCCL_PLUGIN_H
#define HCCL_PLUGIN_H

#include <string>
#include "hccl_common_defs.h"
#include <json.hpp>
#include <mutex>

enum class PLUGIN_MESSAGE_TYPE {
    BROADCAST = 0,
    COMMAND,
    MESSAGE
};
// 插件基础接口
class HcclPlugin {

public:

    static const int MAX_SCAN_DEPTH;    // 最大扫描深度
    static const std::string PLUGIN_PATH;
    static const std::string MANIFEST_FILE;

    struct Manifest {
        static const std::string pluginName;
        static const std::string pluginVersion;
        static const std::string pluginEntry;
        struct pluginDependency {
            static const std::string hostVersion;
        };
    };

    struct PluginMessage {
        static const std::string messageType;
        static const std::string messageAction;
        static const std::string messagePayload;
    };

    HcclPlugin(const std::string& pluginPath);
    ~HcclPlugin();
    // 禁止拷贝，防止文件描述符重复关闭逻辑混乱
    HcclPlugin(const HcclPlugin&) = delete;
    HcclPlugin& operator=(const HcclPlugin&) = delete;

    // 启动/停止插件
    HcclSim::HcclVmResult Start();
    HcclSim::HcclVmResult Stop();
    // 发送命令
    HcclSim::HcclVmResult SendMessage(PLUGIN_MESSAGE_TYPE type, 
                                        const std::string& action, 
                                        const nlohmann::json& payload = nlohmann::json::object());

    int32_t GetPid() const { return m_pid; }
    int32_t GetStdinFd() const { return m_stdinFd; }
    bool IsRunning() const;
    std::string GetTag() const;
    

private:
    std::vector<char*> PrepareArgs(const std::string& command);
    std::string m_pluginPath;
    nlohmann::json m_manifest;
    int32_t m_pid;
    int32_t m_stdinFd;
    std::mutex m_mutex;
};

#endif // HCCL_PLUGIN_H