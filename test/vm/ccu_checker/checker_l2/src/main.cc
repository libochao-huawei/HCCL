#include "hccl_vm_subcmd.h"
#include "hccl_common_defs.h"
#include "hccl_vm_log.h"

#include <CLI11.hpp>

#include <boost/interprocess/ipc/message_queue.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>

using namespace HcclSim;
namespace ipc = boost::interprocess;
namespace fs = std::filesystem;

bool g_singleExe {false};
bool g_bashFlag {false};
std::uint32_t g_level {2};

const std::string HVM_ENV_KEY = "_HVM_ENV_PATH";
const std::string g_binDir = GetBinLocation();
// 定义队列名称和大小配置
const char* MQ_REQ_NAME = "host_mq_request";
const char* MQ_RESP_NAME = "host_mq_response";
const int MAX_MSG_SIZE = 4096; // 单条命令最大长度
const int MAX_MSG_COUNT = 5; // 队列深度

void ParseCommand(std::string& cmd) {
    CLI::App app{"Hccl模拟器"};
    app.set_help_all_flag("--help-all", "展示所有子命令help");

// start子命令
    auto sub_start = app.add_subcommand("start", "start: 启动仿真环境,请勿在子bash中重复启用");
    std::string configFileName;
    TopoMeta topoMeta;

    sub_start->add_option("configFile", configFileName, "加载建模配置yaml文件")->required()->check(FileInModelDir);
    sub_start->add_option("--level", g_level, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");

    sub_start->callback([&]() {
        if (g_bashFlag) {
            HCCL_VM_WARN("[HVM] hccl-vm 已经启动, 请勿在子bash中再次启动");
            return;
        }
        HCCL_VM_INFO("[HVM] Initializing: Model={}, Level={}", configFileName, g_level);
        std::vector<std::string> serverIdx2Ip;
        if(!ParseYamlTopo(configFileName, topoMeta, serverIdx2Ip)) return;
        auto ret = InitHvmEnv(topoMeta, serverIdx2Ip, g_level);
        if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
            HCCL_VM_ERROR("[HVM] 初始化模拟环境失败，正在清理环境");
            auto cleanRet = HcclVmExit();
            if (cleanRet != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
                HCCL_VM_ERROR("[HVM] 清理环境失败，请检查环境残留");
            }
            return;
        }
        return;
    });
    // start : oneShot模式
    auto sub_oneShot = app.add_subcommand("run", "算例one_shot运行模式, 请勿在子bash中重复启用,命令后必须接有算例执行指令");
    sub_oneShot->add_option("configFile", configFileName, "加载建模配置yaml文件")->required()->check(FileInModelDir);
    sub_oneShot->add_option("--level", g_level, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");
    sub_oneShot->allow_extras(true);
    sub_oneShot->callback([&]() {
        if (g_bashFlag) {
            HCCL_VM_WARN("[HVM] hccl-vm 已经启动, 请勿在子bash中one_shot运行算例,请退出子bash再尝试");
            return;
        }
        g_singleExe = true;
        std::vector<std::string> leftargvs = sub_oneShot->remaining();
        if (leftargvs.empty()) {
            HCCL_VM_ERROR("[HVM] one_shot模式, 必须接有算例运行指令");
            return;
        }
        HCCL_VM_INFO("[HVM] Initializing: Model={}, Level={}", configFileName, g_level);
        std::vector<std::string> serverIdx2Ip;
        if(!ParseYamlTopo(configFileName, topoMeta, serverIdx2Ip)) return;
        auto ret = InitHvmEnv(topoMeta, serverIdx2Ip, g_level);
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
        std::string proxyPath = g_binDir + "/libhccl_proxy_level2.so";
        setenv("LD_PRELOAD", proxyPath.c_str(), 1);
        int sysRet = std::system(syscmd.cmd().c_str()); // system() 会阻塞当前进程直到子命令结束
        if (sysRet != 0) {
            HCCL_VM_ERROR("[HVM] 算例执行失败: {}", sysRet);
        }
        RemoveFromLDPreload(proxyPath);
        auto cleanRet = HcclVmExit();
        return;
    });
// model 子命令 // todo
    auto sub_model = app.add_subcommand("model", "管理建模文件");
    sub_model->require_subcommand(1);
    auto model_list = sub_model->add_subcommand("list", "展示建模文件");
    model_list->callback([&]() {
        g_singleExe = true;
        ShowModel();
    });
// plugin 子命令  // todo
    std::string plugName;
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
        g_singleExe = true;
        auto ret = RunUserPlugin(plugName);
    });
    // list
    auto plugin_list = sub_plugin->add_subcommand("list", "展示已安装插件");
    plugin_list->callback([&]() {
        g_singleExe = true;
        auto ret = ShowUserPlugin();
    });

    try {
        app.parse(cmd, true); // true 表示命令第一个字段为程序名
    } catch (const CLI::ParseError &e) { // 非help请求，强行打印help
        if (e.get_exit_code() != 0) {
            std::cerr << "[HVM] [ERROR] 参数错误，请参考:\n\n" << app.help() << std::endl;
        }
        if (g_bashFlag) {
            app.exit(e);
        } else {
            std::exit(app.exit(e));
        }
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
        HCCL_VM_INFO("[HVM] HOST : 收到命令解析请求: {}", cmd);

        // 执行接收到的命令逻辑
        ParseCommand(cmd);
        std::string resp = "Success SubCommand Received";
        // 发送回执, 简易回执
        mq_resp.send(resp.data(), resp.size(), 0);
    }
}

void StartHostServer(int argc, char *argv[]) {
    std::string binDir = GetBinLocation();
    std::string cmd = ArgvToString(argc, argv);
    if (argc == 1) cmd += " --help";
    ParseCommand(cmd);
    if (g_singleExe) return;

    // 初始化HOST通信队列
    ipc::message_queue::remove(MQ_REQ_NAME);
    ipc::message_queue::remove(MQ_RESP_NAME);
    // 启动监听线程
    std::thread server(ServerListen);
    server.detach();

    // Fork Bash
    pid_t pid = fork();
    g_bashFlag = true;
    if (pid == -1) {
        perror("fork failed");
        exit(1);
    } else if (pid == 0) {
        // Child Process(Bash)
        // 劫持库存在性判断
        std::string hcclVmbin = binDir + "/hccl-vm";
        std::string proxyPath = binDir + "/libhccl_proxy_level" + std::to_string(g_level) + ".so";
        if (!fs::exists(proxyPath)) {
            std::cerr << "[HVM] [ERROR] proxy 劫持库不存在 " << proxyPath << " 请检查proxy劫持库:" << std::endl;
            std::cerr << "1.劫持库是否build和install成功. 2.模拟等级level是否和proxy 劫持库版本一致, 当前模拟等级: ";
            std::cerr << g_level << " 默认模拟等级: 1" << std::endl;
            return;
        }

        // 设置环境变量，让子进程里的工具知道处于子bash
        setenv(HVM_ENV_KEY.c_str(), binDir.c_str(), 1);
        std::string bashrcHack = 
            "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi;\n"
            "alias hccl-vm=\"" + hcclVmbin + "\";\n"
            "PS1=\"(hvm)$> \";\n";
        int pipefds[2];
        if (pipe(pipefds) == -1) {
            HCCL_VM_ERROR("[HVM] pipe failed");
            exit(1);
        }
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
    char buffer[MAX_MSG_SIZE];
    ipc::message_queue::size_type recvd_size;
    unsigned int priority;
    mq_resp.receive(&buffer, MAX_MSG_SIZE, recvd_size, priority);
    HCCL_VM_INFO("[HVM] Client : 收到命令解析响应: {}", std::string(buffer, recvd_size));
}

int main(int argc, char *argv[])
{
    LogConfig config;
    config.consoleLevel = 0;
    InitLogger(config);

    const char* envCheck = std::getenv(HVM_ENV_KEY.c_str());

    if (envCheck != nullptr) {
        StartHostClient(argc, argv);
    } else {
        StartHostServer(argc, argv);
    }
    return 0;
}