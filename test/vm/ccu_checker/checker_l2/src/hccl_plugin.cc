
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <fstream>
#include <thread>
#include "hccl_plugin.h"
#include "hccl_common_defs.h"
#include "hccl_common_macro.h"
#include "hccl_vm_log.h"
#include <json.hpp>

using namespace HcclSim;

const int HcclPlugin::MAX_SCAN_DEPTH = 2;    // 最大扫描深度
const std::string HcclPlugin::PLUGIN_PATH = "/plugin";
const std::string HcclPlugin::MANIFEST_FILE = "/manifest.json";

const std::string HcclPlugin::Manifest::pluginName    = "name";
const std::string HcclPlugin::Manifest::pluginVersion = "version";
const std::string HcclPlugin::Manifest::pluginEntry   = "entry";

const std::string HcclPlugin::Manifest::pluginDependency::hostVersion = "min_core_version";

const std::string HcclPlugin::PluginMessage::messageType = "type";
const std::string HcclPlugin::PluginMessage::messageAction = "action";
const std::string HcclPlugin::PluginMessage::messagePayload = "payload";

HcclPlugin::HcclPlugin(const std::string& pluginPath)
{
    m_pluginPath = pluginPath;
    std::string manifestPath = pluginPath + HcclPlugin::MANIFEST_FILE;
    std::ifstream file(manifestPath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Manifest not found at: " + manifestPath);
    }

    try {
        // 直接以 JSON 形式读入整个文件
        file >> m_manifest;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }
    HcclVmResult ret = Start();

    // 立即启动
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        throw std::runtime_error("Failed to launch plugin: " + GetTag());
    }
}

HcclPlugin::~HcclPlugin()
{
    Stop();
}

HcclVmResult HcclPlugin::Start()
{
    std::string entryCmd = m_manifest.value(HcclPlugin::Manifest::pluginEntry, "");
    if (entryCmd.empty()) {
        HCCL_VM_INFO("[HcclPlugin::{}] Empty Entry Command", __func__);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    HCCL_VM_INFO("[HcclPlugin::{}] Entry Command: {}", __func__, entryCmd);
    int32_t fds[2];
    if (pipe(fds) == -1) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    m_pid = fork();
    if (m_pid < 0) {
        ::close(fds[0]);
        ::close(fds[1]);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    if (m_pid == 0) {
        // 子进程逻辑
        prctl(PR_SET_PDEATHSIG, SIGTERM);   // 父进程退出时自动退出子进程

        ::close(fds[1]); // 关闭不需要的写端

        // 将管道读端重定向到标准输入 (STDIN)
        if (dup2(fds[0], STDIN_FILENO) == -1) {
            _exit(EXIT_FAILURE);
        }
        ::close(fds[0]);

        // 切换到插件工作目录，确保相对路径有效
        if (chdir(m_pluginPath.c_str()) != 0) {
            _exit(EXIT_FAILURE);
        }

        // 准备参数并执行
        auto argv = PrepareArgs(entryCmd);
        execvp(argv[0], argv.data());
        perror("execvp");
        // 如果 execvp 执行成功，代码不会走到这里
        std::_Exit(EXIT_FAILURE);
    } else {
        // 父进程逻辑
        ::close(fds[0]); // 关闭不需要的读端
        m_stdinFd = fds[1];

        // 建议：设置管道写端为非阻塞模式，防止 PostMessage 时因插件不读取而卡死主程序
        int flags = fcntl(m_stdinFd, F_GETFL, 0);
        fcntl(m_stdinFd, F_SETFL, flags | O_NONBLOCK);

        std::this_thread::sleep_for(std::chrono::milliseconds(5)); 

        int status = 0;
        pid_t res = waitpid(m_pid, &status, WNOHANG);

        if (res == 0) {
            // 情况 A: 子进程正常运行中（这是我们预期的）
            return HcclVmResult::HCCL_SIM_SUCCESS;
        } else if (res == m_pid) {
            // 情况 B: 子进程已经退出，通常是 execvp 失败
            // 检查退出码，如果是通过 _exit(EXIT_FAILURE) 退出的
            if (WIFEXITED(status)) {
                int exitCode = WEXITSTATUS(status);
                HCCL_VM_ERROR("[HcclPlugin::{}] Plugin [{}] failed to start with code [{}].", __func__, GetTag(), exitCode);
            }

            // 清理现场，防止后续逻辑误以为子进程还在
            ::close(m_stdinFd);
            m_stdinFd = -1;
            m_pid = -1; 
            return HcclVmResult::HCCL_SIM_E_INTERNAL; // 返回具体的错误枚举
        } else {
            // 情况 C: waitpid 出错
            ::close(m_stdinFd);
            m_stdinFd = -1;
            m_pid = -1;
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

}

HcclVmResult HcclPlugin::Stop()
{
    if (m_pid <= 0) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    int status;
    pid_t result = waitpid(m_pid, &status, WNOHANG);

    if (result == m_pid || (result == -1 && errno == ECHILD)) {
        // 子进程已经提前挂了，我们只需要清理本地句柄
        m_pid = -1;
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stdinFd != -1) {
            ::close(m_stdinFd);
            m_stdinFd = -1;
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // 1. 发送stop指令 (协议层停止)
    SendMessage(PLUGIN_MESSAGE_TYPE::COMMAND, "stop");

    // 2. 关闭管道写端 (这会让子进程在读 stdin 时收到 EOF)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stdinFd != -1) {
            ::close(m_stdinFd);
            m_stdinFd = -1;
        }
    }

    // 3. 进入 5 秒等待期
    bool exited = false;
    auto start = std::chrono::steady_clock::now();
    
    // 使用 1 秒一次的频率进行非阻塞检查 (满足你之前的 sleep(1) 想法)
    while (true) {
        int32_t status;
        pid_t result = waitpid(m_pid, &status, WNOHANG);

        if (result == m_pid || (result == -1 && errno == ECHILD)) {
            exited = true;
            break;
        }

        // 检查是否超过 5 秒
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= 5) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 4. 超时后不强制 kill，而是输出告警信息
    if (!exited) {
        HCCL_VM_ERROR("[HcclPlugin::{}] Plugin [{}] detected as not exiting normally.", __func__, GetTag());
        HCCL_VM_ERROR("[HcclPlugin::{}] [ACTION REQUIRED] Please manually check or terminate PID: {:d}", __func__, m_pid);
        
        // 既然无法回收，我们将该 PID 记录在日志后放弃管理
        // 防止析构函数再次产生误判
    } else {
        HCCL_VM_INFO("[HcclPlugin::{}] [{}] exited successfully.", __func__, GetTag());
    }

    m_pid = -1;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclPlugin::SendMessage(PLUGIN_MESSAGE_TYPE type, 
                                        const std::string& action, 
                                        const nlohmann::json& payload)
{
    if (m_pid <= 0 || m_stdinFd == -1) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // 1. 构造精简协议：t (type), a (action), p (payload)
    nlohmann::json msg;
    msg[HcclPlugin::PluginMessage::messageType] = static_cast<int32_t>(type);
    msg[HcclPlugin::PluginMessage::messageAction] = action;
    
    // 即使 payload 为空也传个空对象 {}，方便插件端解析
    msg[HcclPlugin::PluginMessage::messagePayload] = payload;

    // 2. 序列化为单行
    std::string packet = msg.dump(-1) + "\n";

    // 3. 物理写入（加锁确保包完整）
    std::lock_guard<std::mutex> lock(m_mutex);
    ssize_t written = ::write(m_stdinFd, packet.c_str(), packet.size());
    if (written == static_cast<ssize_t>(packet.size())) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } else {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

}

bool HcclPlugin::IsRunning() const {
    if (m_pid <= 0) {
        return false;
    }

    int32_t status;
    // WNOHANG: 如果子进程没结束，立即返回 0
    pid_t result = waitpid(m_pid, &status, WNOHANG);

    if (result == 0) {
        // 子进程正在运行
        return true;
    } else if (result == m_pid) {
        // 子进程已经退出
        return false;
    } else {
        // 可能是发生了错误（例如子进程已被回收）
        return false;
    }
}

std::string HcclPlugin::GetTag() const {
    return m_manifest.value(HcclPlugin::Manifest::pluginName, "Unknown");
}


std::vector<char*> HcclPlugin::PrepareArgs(const std::string& command) {
    std::vector<char*> argv;
    
    // 1. 在局部创建一个副本（每个子进程独立，无需 static）
    // 使用 std::vector<char> 以确保内存连续且可写
    std::vector<char> buffer(command.begin(), command.end());
    buffer.push_back('\0');

    // 2. 使用 strtok_r (线程安全版本)
    char* saveptr = nullptr;
    // 注意：这里的 buffer.data() 指向的是子进程栈/堆上的副本
    char* token = strtok_r(buffer.data(), " ", &saveptr);
    
    while (token != nullptr) {
        // 这里有一个关键点：token 指向 buffer 内部
        // 我们需要把这个 token 真正拷贝出来，否则 buffer 销毁后指针就失效了
        argv.push_back(strdup(token)); 
        token = strtok_r(nullptr, " ", &saveptr);
    }
    
    argv.push_back(nullptr);
    return argv;
}