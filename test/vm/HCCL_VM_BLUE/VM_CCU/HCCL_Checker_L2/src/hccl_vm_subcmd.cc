#include "hccl_vm_subcmd.h"
#include "yaml-cpp/yaml.h"
#include "hccl_data_dump.h"
#include "gen_device_topofile.h"
#include "hccl_vm_log.h"

using namespace HcclSim;
namespace fs = std::filesystem;

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

bool ParseYamlTopo(std::string& fileName, TopoMeta& topo) {
    try {
        std::string filePath = GetBinLocation() + "/cluster_model/" + fileName + ".yaml";
        YAML::Node root = YAML::LoadFile(filePath);

        if (!root["meta"]) {
            std::cerr << "[HVM] [ERROR] YAML : 'meta' node not found." << std::endl;
            return false;
        }
        uint32_t podNum = root["meta"]["podNum"].as<uint32_t>();
        uint32_t serNum = root["meta"]["serNum"].as<uint32_t>();
        uint32_t rankNum = root["meta"]["rankNum"].as<uint32_t>();
        HCCL_VM_DEBUG("PodNum: {}, SerNum: {}, RankNum: {}", podNum, serNum, rankNum);
        topo.reserve(podNum);
        if (podNum <= 0 || podNum >1024 || serNum <= 0 || serNum >1024 || rankNum <= 0 || rankNum >1024) {
            std::cerr << "[HVM] [ERROR] YAML : 'meta' number not surport, please check your config.yaml." << std::endl;
            return false;
        }
        if (root["topology"] && root["topology"].IsSequence()) {
            for (const auto& pod : root["topology"]) {
                SuperPodMeta superPodMeta;
                if (pod["podId"] && pod["servers"] && pod["servers"].IsSequence()) {
                    for (const auto& ser : pod["servers"]) {
                        ServerMeta serverMeta;
                        if (ser["serId"] && ser["ranks"] && ser["ranks"].IsSequence()) {
                            serverMeta = ser["ranks"].as<std::vector<uint32_t>>();
                        }
                        superPodMeta.push_back(serverMeta);
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
            std::cerr << "[HVM] [ERROR] YAML : Pod count mismatch! Meta: " << podNum
                      << ", Actual: " << checkPodNum << std::endl;
            isSuccess = false;
        }
        if (checkSerNum != serNum) {
            std::cerr << "[HVM] [ERROR] YAML : Pod count mismatch! Meta: " << serNum
                      << ", Actual: " << checkSerNum << std::endl;
            isSuccess = false;
        }
        if (checkRankNum != rankNum) {
            std::cerr << "[HVM] [ERROR] YAML : Pod count mismatch! Meta: " << rankNum 
                      << ", Actual: " << checkRankNum << std::endl;
            isSuccess = false;
        }
        return isSuccess;
    } catch (const YAML::Exception& e) {
        std::cerr << "[HVM] [ERROR] Error parsing YAML: " << e.what() << std::endl;
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
        std::cerr << "[HVM] [ERROR] path not exist -> " << modelPath << std::endl;
        return;
    }
    if (!fs::is_directory(modelPath)) {
        std::cerr << "[HVM] [ERROR] path not a dict -> " << modelPath << std::endl;
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
        std::cerr << "[HVM] [INFO] there is no model" << std::endl;
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

HcclVmResult InitHvmEnv(TopoMeta topoMeta, uint32_t level) {
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
    topoGen.Init(topoMeta, "");

    std::string checkerTag = "checker";
    std::string dumperTag =  "dumper";

    auto chkInstallRet = InstallUserPlugin(checkerTag);
    auto dmpInstallRet = InstallUserPlugin(dumperTag);
    if (chkInstallRet != HCCL_SIM_HOST_SUCCESS_CMD || dmpInstallRet != HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("[HVM] default plugin install fail, please check your plugin path");
    }
    std::cout << "======================================"<< std::endl;
    auto showPluginRet = ShowUserPlugin();
    std::cout << "如果想运行runner模式,请安装librunner.so插件. ";
    std::cout << "注意: 安装librunner.so将卸载checker插件" << std::endl;
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
            std::cerr << "[ERROR] plugin Uninstall fail : " << pluginTags[i] <<std::endl;
            return rets[i];
        }
    }

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult ShowUserPlugin() {
    std::vector<std::string> listPlugins{};
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    listPlugins = pluginManager.GetPluginStatus();

    if (listPlugins.empty()) {
        std::cout << "no plugin installed in hccl_vm" <<std::endl;
    } else {
        for (auto &plugin : listPlugins) {
            std::cout << plugin << std::endl;
        }
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}