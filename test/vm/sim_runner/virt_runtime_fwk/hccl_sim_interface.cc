/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim interface
 */

#include "hccl_sim_interface.h"
#include "hccl_sim_pub_stub.h"
#include "rts_stub.h"
#include "stub_rank_table.h"
#include <thread>
#include <iostream>
#include <vector>
#include <signal.h>
#include <execinfo.h>
#include <cstring>
#include "hccl.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <unistd.h>
#include "llt_hccl_stub_pub.h"
#include <atomic>
#include <iostream>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>
#include "phy_topo.h"
#define private public
#include "env_config.h"
#undef private
#include "SimRunnerMgr.h"
#include "hccl_sim_op_flow.h"
#include <sstream>

using namespace std;

void SignalHandler(int sig)
{
    printf("received signal %d !!!\n", sig);
    void *buffer[30];
    int size = backtrace(buffer, 30);  // 获取当前调用栈地址
    char **symbols = backtrace_symbols(buffer, size);  // 将地址转换为可读字符串
    for (int i = 0; i < size; i++) {
        printf("%s\n", symbols[i]);
    }
    free(symbols);
    exit(EXIT_FAILURE);
}

void HcclSimInterface::SetEnv()
{
    for (const auto &cfg : situation.GetEnv()) {
        setenv(cfg.first.c_str(), cfg.second.c_str(), 1);
    }

    // 根据环境变量更新loglevel
    auto logLevelEnv = getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (logLevelEnv != nullptr) {
        auto logLevel = atoi(logLevelEnv);
        log_level_set_stub(logLevel);
    }

    // 设置Ca Model场景使用标识
    auto caModeleEnv = getenv("CCU_INS_USE_SCENE");
    if (caModeleEnv != NULL && !strcmp(caModeleEnv, "CA_MODEL")) {
        caModelFlag_ = true;
         std::cout << "HcclSimInterface::SetEnv()" << caModeleEnv << (int)caModelFlag_ << std::endl;
    }
     std::cout << "HcclSimInterface::SetEnv()" << caModeleEnv << (int)caModelFlag_ << std::endl;
    auto ccuVersionEnv = getenv("CcuVersion");
    if (ccuVersionEnv != NULL && !strcmp(ccuVersionEnv, "CCU_V2")) {
        std::cout << "HcclSimInterface::SetEnv()" << (int)ccuVersionFlag_ << std::endl;
        ccuVersionFlag_ = CcuVersion::CCU_V2;
        std::cout << "HcclSimInterface::SetEnv() CCU_V2" << (int)ccuVersionFlag_ << std::endl;
    } else {
        ccuVersionFlag_ = CcuVersion::CCU_V1;
        std::cout << "HcclSimInterface::SetEnv() CCU_V1" << (int)ccuVersionFlag_ << std::endl;
    }

    auto mc2Env = getenv("MC2_ENABLE");
    if (mc2Env != NULL && (atoi(mc2Env) == 1)) {
        mc2Flag_ = true;
    }
}

void HcclSimInterface::UnsetEnv()
{
    for (const auto &cfg : situation.GetEnv()) {
        unsetenv(cfg.first.c_str());
    }
}

bool createSingleVarable(Situation &situation, bool isAicpu, bool isCcu, CcuVersion ccuVersionFlag, bool caModelFlag , bool mc2Flag)
{
    int rankSize = situation.GetRankSize();
    auto shmPoolMgr = std::make_shared<ShmPoolManager>(isAicpu, rankSize);
    if (shmPoolMgr->CreateShmPool(situation) == false) {
        return false;
    }

    SimRunnerMgr::GetInstance().Init(rankSize, isAicpu, isCcu, ccuVersionFlag, caModelFlag, mc2Flag, shmPoolMgr);
    if (CreateDeviceProcesses(rankSize) == false) {
        HCCL_ERROR("[AICPU][Start][HcclSimInterface]create_device_processes failed");
        return false;
    }
    return true;
}

void deleteSingleVarable()
{
    SimRunnerMgr::GetInstance().Destory();
}

void deleteFilesInDirectory(const std::string &directory)
{
    DIR *dir = opendir(directory.c_str());
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                std::string filePath = directory + "/" + entry->d_name;
                struct stat fileStat;
                if (stat(filePath.c_str(), &fileStat) == 0 && S_ISREG(fileStat.st_mode)) {
                    if (unlink(filePath.c_str()) == 0) {
                        std::cout << "已删除文件: " << filePath << std::endl;
                    } else {
                        std::cerr << "删除文件失败: " << filePath << std::endl;
                    }
                }
            }
        }
        closedir(dir);
    } else {
        std::cerr << "无法打开目录: " << directory << std::endl;
    }
}

static const std::map<HcclDataType, string> DATA_STRING_MAP = {
    {HcclDataType::HCCL_DATA_TYPE_INT8, "INT8"},
    {HcclDataType::HCCL_DATA_TYPE_INT16, "INT16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "INT32"},
    {HcclDataType::HCCL_DATA_TYPE_FP16, "FP16"},
    {HcclDataType::HCCL_DATA_TYPE_FP32, "FP32"},
    {HcclDataType::HCCL_DATA_TYPE_INT64, "INT64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "UINT64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, "UINT8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "UINT16"},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, "UINT32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "FP64"},
    {HcclDataType::HCCL_DATA_TYPE_BFP16, "BFP64"}
};

static const std::map<HcclReduceOp, string> REDUCEOP_STRING_MAP = {
    {HcclReduceOp::HCCL_REDUCE_SUM, "SUM"},
    {HcclReduceOp::HCCL_REDUCE_PROD, "PROD"},
    {HcclReduceOp::HCCL_REDUCE_MAX, "MAX"},
    {HcclReduceOp::HCCL_REDUCE_MIN, "MIN"},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, "INVALID_ReduceOp"}
};

static const std::map<CcuVersion, string> CCU_VERSION_STRING_MAP = {
    {CcuVersion::CCU_INVALID, "INVALID_CcuVersion"},
    {CcuVersion::CCU_V1, "CCU_V100"},
    {CcuVersion::CCU_V2, "CCU_V121"}
};

bool HcclSimInterface::CreateCcuInfoDir()
{
    if (!ccuFlag_) {
        return true;  // 非CCU模式不创建目录
    }
    auto ccuPrimName = getenv("PRIM_QUEUE_GEN_NAME");
    if (ccuPrimName == NULL ) {
        HCCL_ERROR("PRIM_QUEUE_GEN_NAME env is empty");
        return true;
    }
    // 判断目录是否存在
    char buffer[512];
    std::cout << "HcclSimInterface::CreateCcuInfoDir()" << (int)ccuVersionFlag_ << std::endl;
    sprintf(buffer, "%s_info_%dp_%s_data[%llu]_%s_ReduceOp_%s",
            CCU_VERSION_STRING_MAP.at(ccuVersionFlag_).c_str(), situation.GetDeviceNum(), ccuPrimName, situation.GetCount(),
            DATA_STRING_MAP.at(situation.GetDataType()).c_str(), REDUCEOP_STRING_MAP.at(situation.GetReduceOp()).c_str());
    string &GeneCcuInfoFilePath = CcuResouceManager::GetInstance().GetCcuInfoFilePath();
    GeneCcuInfoFilePath = buffer;
    if (mc2Flag_) {
        GeneCcuInfoFilePath = "mc2_" + GeneCcuInfoFilePath;
    }
    HCCL_INFO("GeneCcuInfoFilePath = %s", GeneCcuInfoFilePath.c_str());
    if (access(GeneCcuInfoFilePath.c_str(), F_OK) == 0) {
        // 删除目录下的所有文件
        deleteFilesInDirectory(GeneCcuInfoFilePath);
    }
    else {
        // 目录不存在，创建目录
        if (mkdir(GeneCcuInfoFilePath.c_str(), 0777) == 0) {
            std::cout << "目录创建成功！" << std::endl;
        } else {
            std::cout << "目录创建失败" << std::endl;
            return false;
        }
    }
    return true;
}

void HcclSimInterface::Start()
{
    if (!InitSimResource()) {
        return;
    }
    std::cout << "===== st test case " << testcaseName << " ===== start =====" << std::endl;
    
    int myRank = 0;
    vector<int> pids;
    for (int serverIndex = 0; serverIndex < situation.GetServerNum(); ++serverIndex) {
        // threads.emplace_back();
        for (int deviceIndex = 0; deviceIndex < situation.GetDeviceNum(); ++deviceIndex) {
            auto *params = new SimParams();
            params->commId = "st-hccl_sim_runner";
            params->situation = situation;
            params->serverId = serverIndex;
            params->deviceId = deviceIndex;
            params->myRank = myRank++;
            params->isAicpu = aicpuFlag_;
            params->mc2Flag = mc2Flag_;
            contexts.push_back(params);
            // fork子进程，每个进程模拟一张卡
            pid_t pid = fork();
            if (pid < 0) {
                perror("fork failed");  // fork失败
                exit(EXIT_FAILURE);
            } else if (pid == 0) {
                InternalProcess(params);
                exit(EXIT_SUCCESS);  // 子进程执行完毕后退出
            } else {
                pids.push_back(pid);  // 父进程记录子进程pid
            }
        }
    }

    // 父进程等待所有子进程执行完毕
    for (size_t i = 0; i < pids.size(); ++i) {
        int status;
        waitpid(pids[i], &status, 0);
    }

    if (aicpuFlag_) {
        for (int serverIndex = 0; serverIndex < situation.GetServerNum(); ++serverIndex) {
            for (int deviceIndex = 0; deviceIndex < situation.GetDeviceNum(); ++deviceIndex) {
                if (SimRunnerMgr::GetInstance().GetPid(deviceIndex) != 0) {
                    ExitSubProcess(deviceIndex);  // 退出Device进程
                }
            }
        }
    }

    // 资源清理：环境遍历、ranktable
    ClearSimResource(situation.GetDevType());
    std::cout << "===== st test case " << testcaseName << " ===== finish =====" << std::endl;
}

bool HcclSimInterface::InitSimResource()
{
    // 每个用例执行前的资源初始化
    SetEnv();  // 先设置环境变量再获取aicpuFlag_
    aicpuFlag_ = situation.IsAicpuAcceleratorMode();
    ccuFlag_ = situation.IsCcuAcceleratorMode();
    if (caModelFlag_) {
        if (!CreateCcuInfoDir()) {
            HCCL_INFO("ccu infor dir create fail");
        }
    }

    DeviceType devType = situation.GetDevType();
    networkMgr_ = std::make_shared<SimNetworkManager>(devType, situation.GetClusterType(), ccuFlag_);
    if (!networkMgr_->RankTableFileCreate()) {
        return false;
    }

    if (createSingleVarable(situation, aicpuFlag_, ccuFlag_, ccuVersionFlag_, caModelFlag_, mc2Flag_) == false) {
        return false;
    }

    // David执行用例时强制解析环境变量，否则还是上个用例的参数(单例类导致)
    if (devType == DeviceType::DEV_TYPE_910_95) {
        // Hccl::EnvConfig::GetInstance().Parse();
    }

    return true;
}
    
void HcclSimInterface::ClearSimResource(DeviceType devType)
{
    UnsetEnv();
    networkMgr_->DelRankTableFile();
    if (devType == DeviceType::DEV_TYPE_910_95) {
        networkMgr_->DelTopoInfoFile();
        networkMgr_->DelDieInfoFile();
        // Hccl::PhyTopo::GetInstance()->Clear();  // 虚拟拓扑单例数据清理
    }
}

bool HcclSimInterface::Verify()
{
    auto caModeleEnv = getenv("CCU_INS_USE_SCENE");
    if (caModeleEnv != NULL && strcmp(caModeleEnv, "CA_MODEL")) {
        return true;
    }
    bool flag = true;
    for (auto ctx : contexts) {
        // 这里理论上只需要知道rank数即可，遍历获取每个rank的verify结果
        if (!SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetVerifyResult(ctx->myRank)) {
            flag = false;
        }
        delete ctx;
    }
    deleteSingleVarable();
    return flag;
}