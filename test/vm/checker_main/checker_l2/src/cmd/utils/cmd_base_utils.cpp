#include <iostream>
#include <sys/wait.h>
#include <thread>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <CLI11.hpp>
#include "cmd_base_utils.h"
#include "yaml-cpp/yaml.h"
#include "hccl_data_dump.h"
#include "gen_device_topofile.h"
#include "hccl_vm_log.h"
#include "cmd_base.h"

using namespace HcclSim;
namespace fs = std::filesystem;
namespace ipc = boost::interprocess;
namespace bpt = boost::posix_time;

const std::string HVM_BASH_ENV_KEY = "_HVM_BASH_ENV_PATH";
const std::string g_binDir = GetBinLocation();
// 定义队列名称和大小配置
const char* MQ_REQ_NAME = "host_mq_request";
const char* MQ_RESP_NAME = "host_mq_response";
const int MAX_MSG_SIZE = 4096; // 单条命令最大长度
const int MAX_MSG_COUNT = 5; // 队列深度

bool g_hcclVmBashFlag{false};
std::uint32_t g_hcclVmLevel{2};

std::string GetBinLocation() {
    std::error_code ec;
    fs::path exePath = fs::read_symlink("/proc/self/exe", ec);
    if (ec) {
        throw std::runtime_error("read_symlink failed: " + ec.message());
    }
    return exePath.parent_path().string();
}

std::string ArgvToString(int argc, char *argv[]) {
    std::string cmd;
    for (int i = 0; i < argc; ++i) {
        std::string s = argv[i];
        // 如果包含空格，两头加引号
        if (s.find(' ') != std::string::npos) {
            cmd += "\"" + s + "\"";
        } else {
            cmd += s;
        }
        if (i < argc - 1) cmd += " ";
    }
    return cmd;
}

void RemoveFromLDPreload(const std::string& targetValue) {
    HCCL_VM_DEBUG("[HVM] 清理LD_PRELOAD环境变量: {}", targetValue);
    const char* curVal = std::getenv("LD_PRELOAD");

    if (curVal == nullptr) {
        return;
    }
    std::string envStr(curVal);
    
    // 如果当前值就是目标值，直接 unset
    if (envStr == targetValue) {
        unsetenv("LD_PRELOAD");
        return;
    }

    std::stringstream ss(envStr);
    std::string item;
    std::string envStrNew;
    bool first = true;

    while (std::getline(ss, item, ':')) {
        // 过滤空项（双冒号情况）和目标项
        if (item.empty() || item == targetValue) {
            continue;
        }
        if (!first) {
            envStrNew += ":";
        }
        envStrNew += item;
        first = false;
    }
    if (envStrNew.empty()) {
        // 如果结果为空，说明只包含要删除的项，直接 unset
        unsetenv("LD_PRELOAD");
    } else {
        // 覆盖原变量 (overwrite = 1)
        setenv("LD_PRELOAD", envStrNew.c_str(), 1);
    }
}

bool ParseYamlTopo(std::string& fileName, TopoMeta& topo, std::vector<std::string> &serverIdx2Ip) {
    try {
        std::string filePath = GetBinLocation() + "/cluster_model/" + fileName + ".yaml";
        YAML::Node root = YAML::LoadFile(filePath);

        if (!root["meta"]) {
            HCCL_VM_ERROR("[HVM] YAML : 'meta' node not found.");
            return false;
        }
        uint32_t podNum = root["meta"]["podNum"].as<uint32_t>();
        uint32_t serNum = root["meta"]["serNum"].as<uint32_t>();
        uint32_t rankNum = root["meta"]["rankNum"].as<uint32_t>();
        HCCL_VM_DEBUG("PodNum: {}, SerNum: {}, RankNum: {}", podNum, serNum, rankNum);
        serverIdx2Ip.resize(serNum);
        topo.reserve(podNum);
        if (podNum <= 0 || podNum >1024 || serNum <= 0 || serNum >1024 || rankNum <= 0 || rankNum >1024) {
            HCCL_VM_ERROR("[HVM] YAML : 'meta' number not surport, please check your config.yaml.");
            return false;
        }
        if (root["topology"] && root["topology"].IsSequence()) {
            for (const auto& pod : root["topology"]) {
                SuperPodMeta superPodMeta;
                if (pod["podId"] && pod["servers"] && pod["servers"].IsSequence()) {
                    for (const auto& ser : pod["servers"]) {
                        ServerMeta serverMeta;
                        std::string ipAddr;
                        uint32_t serverId = 0;
                        if (ser["serId"] && ser["ip"] && ser["ranks"] && ser["ranks"].IsSequence()) {
                            serverId = ser["serId"].as<uint32_t>();
                            serverMeta = ser["ranks"].as<std::vector<uint32_t>>();
                            ipAddr = ser["ip"].as<std::string>();
                        }
                        superPodMeta.push_back(serverMeta);
                        serverIdx2Ip[serverId] = ipAddr;
                    }
                }
                topo.push_back(superPodMeta);
            }
        }
        // 检查用户yaml配置是否异常
        uint32_t checkPodNum{0};
        uint32_t checkSerNum{0};
        uint32_t checkRankNum{0};
        bool isSuccess{true};

        checkPodNum = topo.size();
        for (const auto& pod : topo) {
            checkSerNum += pod.size();
            for (const auto& server : pod) {
                checkRankNum += server.size();
            }
        }
        if (checkPodNum != podNum) {
            HCCL_VM_ERROR("[HVM] YAML : Pod count mismatch! Meta: {}, Actual: {}", podNum, checkPodNum);
            isSuccess = false;
        }
        if (checkSerNum != serNum) {
            HCCL_VM_ERROR("[HVM] YAML : Server count mismatch! Meta: {}, Actual: {}", serNum, checkSerNum);
            isSuccess = false;
        }
        if (checkRankNum != rankNum) {
            HCCL_VM_ERROR("[HVM] YAML : Rank count mismatch! Meta: {}, Actual: {}", rankNum, checkRankNum);
            isSuccess = false;
        }
        return isSuccess;
    } catch (const YAML::Exception& e) {
        HCCL_VM_ERROR("[HVM] Exception when parsing YAML: {}", e.what());
        return false;
    }
}

std::string FileInModelDir(const std::string& fileName) {
    std::string filePath = GetBinLocation() + "/cluster_model/" + fileName + ".yaml";
    auto fileExistd = [&]()->bool {
        std::ifstream f(filePath.c_str());
        return f.good();
    };
    if(fileExistd()) {
        return "";
    } else {
        return "[HVM] model File not found: " + filePath; 
    }
}

void ShowModel() {
    std::string modelPath = GetBinLocation() + "/cluster_model";
    if (!fs::exists(modelPath)) {
        HCCL_VM_ERROR("[HVM] path not exist -> {}", modelPath);
        return;
    }
    if (!fs::is_directory(modelPath)) {
        HCCL_VM_ERROR("[HVM] path not a dict -> {}", modelPath);
        return;
    }
    bool hasFiles = false;
    std::cout << "model : " << std::endl;
    for (const auto& entry : fs::directory_iterator(modelPath)) {
        // 过滤：只关心“常规文件”，忽略子文件夹
        if (entry.is_regular_file()) {
            hasFiles = true;
            fs::path filePath = entry.path();
            std::cout << "  " << filePath.stem().string() << "  [description] : ";
            YAML::Node root = YAML::LoadFile(filePath);
            uint32_t podNum = root["meta"]["podNum"].as<uint32_t>();
            uint32_t serNum = root["meta"]["serNum"].as<uint32_t>();
            uint32_t rankNum = root["meta"]["rankNum"].as<uint32_t>();
            std::cout << "PodNum: " << podNum << ", SerNum: " << serNum << ", RankNum: " << rankNum << std::endl;
        }
    }
    if (!hasFiles) {
        HCCL_VM_WARN("[HVM] there is no model");
    }
    return;
}

HcclVmResult HvmInitSHM(TopoMeta topoMeta)
{
    HcclVmResult ret = InitSharedMemory(topoMeta);
    if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_DEBUG("[HVM] Init SharedMemory Success.");
    }
    else {
        HCCL_VM_ERROR("[HVM] Init SharedMemory Fail.");
        return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult HvmInitIPC(TopoMeta topoMeta) {
    uint32_t ipcRankNum = topoMeta.size() * topoMeta[0].size() * topoMeta[0][0].size();
    HcclVmResult ret = InitIpc(ipcRankNum);
    if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_DEBUG("[HVM] Init IPC Success.");
    }
    else {
        HCCL_VM_ERROR("[HVM] IPC Fail.");
        return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

void EnvInit()
{
    // RunnerDB共享内存初始化
    ShmEnvInit();
}

HcclVmResult InitHvmEnv(TopoMeta topoMeta, const std::vector<std::string>& serverIdx2Ip, uint32_t level) {
    EnvInit();
    // 启动仿真环境
    auto shmInitRet = HvmInitSHM(topoMeta);
    auto ipcInitRet = HvmInitIPC(topoMeta);
    if (shmInitRet != HCCL_SIM_HOST_SUCCESS_CMD || ipcInitRet != HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("[HVM] Init Share Memory fail ");
        return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
    }

    // todo: 区分L1和L2
    DeviceTopoGenerator topoGen;
    topoGen.Init(topoMeta, serverIdx2Ip, "");

    std::string checkerTag = "checker";
    std::string dumperTag =  "dumper";

    auto chkInstallRet = InstallUserPlugin(checkerTag);
    auto dmpInstallRet = InstallUserPlugin(dumperTag);
    if (chkInstallRet != HCCL_SIM_HOST_SUCCESS_CMD || dmpInstallRet != HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("[HVM] default plugin install fail, please check your plugin path");
    }
    std::cout << "======================================"<< std::endl;
    ShowUserPlugin();

    std::cout << "======================================="<< std::endl;

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}


HcclVmResult HcclVmExit() {
    HCCL_VM_INFO("start Destroy SharedMemory.");
    SHMManager::DestroyShm();
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    auto ret = pluginManager.StopAllPlugins();
    
    return ret;
}

HcclVmResult InstallUserPlugin(std::string argStr) {
    // 处理插件tag和路径
    HcclVmResult ret {HcclVmResult::HCCL_SIM_HOST_ERROR_CMD};
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);

    // 注册插件
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    ret = pluginManager.RegisterPlugin(argStr);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        return ret;
    }

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult RunUserPlugin(std::string argStr) {
    HcclVmResult ret = DumpData();
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("DumpData failed");
        return ret;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult UninstallUserPlugin(std::string argStr) {
    std::vector<std::string> pluginTags{};

    size_t start = 0;
    size_t end = argStr.find(',');
    while (end != std::string::npos) {
        std::string tag = argStr.substr(start, end - start);
        tag.erase(tag.begin());
        pluginTags.push_back(tag);
        start = end + 1;
        end = argStr.find(',', start);
    }
    std::string lastTag = argStr.substr(start);
    lastTag.erase(lastTag.begin());
    pluginTags.push_back(lastTag);

    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    auto rets = pluginManager.StopPlugins(pluginTags);
    for (int i = 0; i < pluginTags.size(); ++i) {
        if (rets[i] != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("[HVM] plugin Uninstall fail : {}", pluginTags[i]);
            return rets[i];
        }
    }

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

void ShowUserPlugin() {
    std::vector<std::string> listPlugins{};
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    listPlugins = pluginManager.GetPluginStatus();

    if (listPlugins.empty()) {
        HCCL_VM_INFO("[HVM] no plugin installed in hccl_vm");
    } else {
        for (auto &plugin : listPlugins) {
            std::cout << plugin << std::endl;
        }
    }
    return;
}

HcclVmResult SetConsoleLogLevel(int level) {
    auto* proxyConfig = SHMManager::GetProxyConfig();
    if (proxyConfig == nullptr) {
        HCCL_VM_ERROR("[HVM] get proxy config failed");
        return HCCL_SIM_HOST_ERROR_CMD;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(proxyConfig->mutex);
        proxyConfig->consoleLogLevel = level;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult SetFileLogLevel(int level) {
    auto* proxyConfig = SHMManager::GetProxyConfig();
    if (proxyConfig == nullptr) {
        HCCL_VM_ERROR("[HVM] get proxy config failed");
        return HCCL_SIM_HOST_ERROR_CMD;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(proxyConfig->mutex);
        proxyConfig->fileLogLevel = level;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult ShowCurrentLogLevel() {
    auto* proxyConfig = SHMManager::GetProxyConfig();
    if (proxyConfig == nullptr) {
        HCCL_VM_ERROR("[HVM] get proxy config failed");
        return HCCL_SIM_HOST_ERROR_CMD;
    }

    const std::map<int, std::string> levelNameMap
        {{0, "trace"},{1, "debug"},{2, "info"},{3, "warn"},{4, "error"},{5, "critical"}};

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(proxyConfig->mutex);
        std::cout << "Current log levels: "
            << levelNameMap.at(proxyConfig->consoleLogLevel) << " (console)"
            << ", "
            << levelNameMap.at(proxyConfig->fileLogLevel) << " (file)"
            << std::endl;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

void StartHostServer() {
    // 初始化HOST通信队列
    ipc::message_queue::remove(MQ_REQ_NAME);
    ipc::message_queue::remove(MQ_RESP_NAME);
    // 启动监听线程
    std::thread server(ServerListen);
    server.detach();

    // Fork Bash
    pid_t pid = fork();
    g_hcclVmBashFlag = true;
    if (pid == -1) {
        perror("fork failed");
        exit(1);
    } else if (pid == 0) {
        // Child Process(Bash)
        // 劫持库存在性判断
        std::string hcclVmbin = g_binDir + "/hccl-vm";
        std::string proxyPath = g_binDir + "/libhccl_proxy_level" + std::to_string(g_hcclVmLevel) + ".so";
        if (!fs::exists(proxyPath)) {
            HCCL_VM_ERROR("[HVM] [ERROR] proxy 劫持库不存在 {}, 请检查proxy劫持库:" 
                "1.劫持库是否build和install成功. 2.模拟等级level是否和proxy 劫持库版本一致, 当前模拟等级: {}, 默认模拟等级: 2"
                ,proxyPath, g_hcclVmLevel);
            return;
        }

        // 设置环境变量，让子进程里的工具知道处于子bash
        setenv(HVM_BASH_ENV_KEY.c_str(), g_binDir.c_str(), 1);

        // 管道处理的用途是隔绝进程终端在std::cout中的残留
        int pipefds[2];
        if (pipe(pipefds) == -1) {
            HCCL_VM_ERROR("[HVM] pipe failed");
            exit(1);
        }
        std::string fdNum = std::to_string(pipefds[0]);
        std::string bashrcHack = 
            "__HVM_SAVED_PATH=\"$PATH\";\n"
            "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi;\n"
            "export PATH=\"$__HVM_SAVED_PATH:$PATH\";\n"
            "alias hccl-vm=\"" + hcclVmbin + "\";\n"
            "PS1='(hvm)\\$> ';\n" 
            "unset __HVM_SAVED_PATH;\n"
            "exec " + fdNum + "<&-;\n"; 

        ssize_t written = write(pipefds[1], bashrcHack.c_str(), bashrcHack.size());
        if (written != (ssize_t)bashrcHack.size()) {
            HCCL_VM_ERROR("[HVM] Failed to write full script to pipe");
        }
        close(pipefds[1]);
        std::string devFdPath = "/dev/fd/" + std::to_string(pipefds[0]);
        setenv("LD_PRELOAD", proxyPath.c_str(), 1);
        char *bashArgv[] = {
            const_cast<char*>("bash"),
            const_cast<char*>("--rcfile"),
            const_cast<char*>(devFdPath.c_str()),
            const_cast<char*>("-i"),
            NULL
        };
        HCCL_VM_INFO("bash pid : {}", getpid());
        execv("/bin/bash", bashArgv);
        perror("bash execv failed");
        exit(1);
    } else {
        // Parent Process (Host)
        // 等待 bash 结束 (阻塞等待，保持Host存活)
        int status;
        waitpid(pid, &status, 0);
        auto ret = HcclVmExit();
        ipc::message_queue::remove(MQ_REQ_NAME);
        ipc::message_queue::remove(MQ_RESP_NAME);
        HCCL_VM_INFO("[HVM] Shell exited. Host shutting down.");
    }
}

void StartHostClient(int argc, char *argv[]) {
    // 打开队列
    ipc::message_queue mq_req(ipc::open_only, MQ_REQ_NAME);
    ipc::message_queue mq_resp(ipc::open_only, MQ_RESP_NAME);

    std::string cmd = ArgvToString(argc, argv);
    if (argc == 1) {
        cmd += " --help";
    }
    // 向主host发生命令行
    mq_req.send(cmd.data(), cmd.size(), 0);
    // 等待回执，简单回执
    bpt::ptime abs_time = bpt::microsec_clock::universal_time() + bpt::milliseconds(5000); // 等2s
    char buffer[MAX_MSG_SIZE];
    ipc::message_queue::size_type recvd_size;
    unsigned int priority;
    bool hasMessage = mq_resp.timed_receive(&buffer, MAX_MSG_SIZE, recvd_size, priority, abs_time);
    if (hasMessage) {
        HCCL_VM_DEBUG("[HVM] Client : 收到命令解析响应: {}", std::string(buffer, recvd_size));
    } else {
        HCCL_VM_WARN("[HVM] Client : 命令解析响应超时, Client退出");
    }
}

void ServerListen() {
    // 创建队列
    ipc::message_queue mq_req(ipc::create_only, MQ_REQ_NAME, MAX_MSG_COUNT, MAX_MSG_SIZE);
    ipc::message_queue mq_resp(ipc::create_only, MQ_RESP_NAME, MAX_MSG_COUNT, MAX_MSG_SIZE);
    HCCL_VM_INFO("[HVM] HOST Server listening...");

    while(true) {
        char buffer[MAX_MSG_SIZE];
        ipc::message_queue::size_type recvd_size;
        unsigned int priority;
        // 阻塞等待请求 (Receive)
        mq_req.receive(&buffer, MAX_MSG_SIZE, recvd_size, priority);
        
        std::string cmd(buffer, recvd_size);
        HCCL_VM_DEBUG("[HVM] HOST : 收到命令解析请求: {}", cmd);

        // 执行接收到的命令逻辑
        ParseCommand(cmd);
        std::string resp = "Success SubCommand Received";
        // 发送回执, 简易回执
        mq_resp.send(resp.data(), resp.size(), 0);
    }
}

void ParseCommand(std::string& cmd) {
    CLI::App app{"Hccl模拟器"};
    app.set_help_all_flag("--help-all", "展示所有子命令help");

    auto commands = CommandRegistry::CreateAll();

    for (auto& command : commands) {
        command->Setup(app);
    }
    try {
        app.parse(cmd, true); // true 表示命令第一个字段为程序名
    } catch (const CLI::ParseError &e) { // 非help请求，强行打印help
        if (e.get_exit_code() != 0) {
            std::cerr << "[HVM] [ERROR] 参数错误，请参考:\n\n" << app.help() << std::endl;
        }
        if (g_hcclVmBashFlag) {
            app.exit(e);
        } else {
            std::exit(app.exit(e));
        }
    }
}