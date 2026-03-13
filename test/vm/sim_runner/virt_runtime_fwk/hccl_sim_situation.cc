/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim situation
 */

#include "hccl_sim_situation.h"
#include "hccl_sim_pub_stub.h"
#include "stub_rank_table.h"
#include <iostream>
#include <map>

const std::map<DeviceType, std::string> DEV_VERSION_MAP = {
    {DeviceType::DEV_TYPE_V80, "Ascend910"},
    {DeviceType::DEV_TYPE_V71, "Ascend910B1"},
    {DeviceType::DEV_TYPE_910_93, "Ascend910_9391"},
    {DeviceType::DEV_TYPE_910_95, "Ascend910_9591"}
};

void SetFakeSocVersion(DeviceType type)
{
    std::string socVersion;
    auto it = DEV_VERSION_MAP.find(type);
    if (it != DEV_VERSION_MAP.end()) {
        socVersion = it->second;
    } else {
        std::cout << "ST only support V80/V71/910_93/910D now" << std::endl;
        throw std::exception();
    }

    SetFakeSocVersionStub(socVersion);
}

Situation::Situation()
{
    InitSituationEnv();
}

void Situation::InitSituationEnv()
{
    // 日志
    this->SetEnv("ASCEND_GLOBAL_LOG_LEVEL", "3");      // 设置全局日志级别为1（调试信息）
    this->SetEnv("ASCEND_SLOG_PRINT_TO_STDOUT", "1");  // 设置将日志输出到标准输出
    this->SetEnv("HCCL_DIAGNOSE_ENABLE", "1");         // 启用HCCL诊断功能
    this->SetEnv("HCCL_ENTRY_LOG_ENABLE", "1");        // 启用入口日志功能
    this->SetEnv("PROFILING_MODE", "false");           // 禁用性能分析模式

    // 网络与通信配置
    this->SetEnv("HCCL_CONNECT_TIMEOUT", "200");           // 设置连接超时为200秒 ok  可以修改
    this->SetEnv("HCCL_IF_IP", "10.10.10.1");              // 设置HCCL接口IP地址
    this->SetEnv("HCCL_IF_BASE_PORT", "50000");            // 设置HCCL基础端口号
    // this->SetEnv("HCCL_SOCKET_IFNAME", "^=eth0,endvnic");  // 设置绑定的网络接口名称，如 eth0 或 endvnic
    this->SetEnv("HCCL_SOCKET_FAMILY", "AF_INET6");        // 设置使用IPv6地址族
    this->SetEnv("HCCL_EXEC_TIMEOUT", "600");              // 设置SQE执行超时为600秒
    //  this->SetEnv("HCCL_NPU_NET_PROTOCOL", "RDMA");  // 设置网络协议为RDMA（用于高效的远程内存访问）  no
    this->SetEnv("HCCL_RDMA_TC", "100");       // 设置RDMA流量类为100
    this->SetEnv("HCCL_RDMA_SL", "3");         // 设置RDMA服务级别为3
    this->SetEnv("HCCL_RDMA_TIMEOUT", "6");    // 设置RDMA超时为6秒
    this->SetEnv("HCCL_RDMA_RETRY_CNT", "5");  // 设置RDMA重试次数为5次
    // HCCL（硬件加速通信层）配置
    this->SetEnv("HCCL_BUFFSIZE", "200");                // 设置缓冲区大小为200字节   开放
    this->SetEnv("HCCL_ALGO", "level0:NA;level1:ring");  // 设置通信算法为level0:NA 和 level1:ring
    this->SetEnv("HCCL_DETERMINISTIC", "false");  // 禁用HCCL的确定性模式 是否开放，确保在多个训练运行中得到相同的结果

    // 不开放
    this->SetEnv("HCCL_INTRA_PCIE_ENABLE", "1");            // 启用PCIE内部通信 总线协议
    this->SetEnv("HCCL_INTRA_ROCE_ENABLE", "0");            // 禁用RoCE协议 禁用rdma    不开放
    this->SetEnv("HCCL_INTER_HCCS_DISABLE", "FALSE");       // 禁用HCCS（互联组件）功能
    this->SetEnv("HCCL_OP_EXPANSION_MODE", "HOST");         // 设置操作扩展模式为HOST
    this->SetEnv("HCCL_OP_BASE_FFTS_MODE_ENABLE", "TRUE");  // 启用HCCL操作的FFT+模式

    // 系统与路径设置
    this->SetEnv("LD_LIBRARY_PATH", "./");  // 设置共享库路径
    // 特殊功能配置
    this->SetEnv("HCCL_DETOUR", "detour:0");  // 禁用绕行模式

    // 2.0使用
    this->SetEnv("PRIM_QUEUE_GEN_NAME", "AllReduceConcurrMesh");
    this->SetEnv("HCCL_TOPO_FILE_PATH", "./topo.json");
    this->SetEnv("CHIP_VERIFY_ORCHESTRATE_WAY", "PRIM");
}

Situation &Situation::SetDataType(HcclDataType dataType1)
{
    this->dataType = dataType1;
    return *this;
}

Situation &Situation::SetReduceOp(HcclReduceOp reduceOp1)
{
    this->reduceOp = reduceOp1;
    return *this;
}

Situation &Situation::SetMc2Flag(bool mc2Flag1)
{
    this->mc2Flag = mc2Flag1;
    std::cout << "Situation::SetMc2Flag(bool mc2Flag1)" << (int)mc2Flag1 << std::endl;
    return *this;
}

Situation &Situation::SetRoot(int root1)
{
    this->root = root1;
    return *this;
}

Situation &Situation::SetOpType(OpType opType1)
{
    this->opType = opType1;
    return *this;
}

Situation &Situation::SetCount(u64 dataCount)
{
    int rankSize = this->GetRankSize();
    if ((this->opType == OpType::ALLTOALL || this->opType == OpType::REDUCESCATTER) && (dataCount % rankSize != 0)) {
        // 非rank整数倍时数据自适应扩充
        dataCount = ((dataCount + (u64)rankSize - 1) / (u64)rankSize) * (u64)rankSize;
    }
    this->count = dataCount;
    return *this;
}

Situation &Situation::SetShmSize(u64 shmSize)
{
    this->shmSize = shmSize;
    return *this;
}

Situation &Situation::SetEnv(const std::string &name, const std::string &value)
{
    this->envConfigs[name] = value;
    return *this;
}

Situation &Situation::addUserEnv(std::map<std::string, std::string> &mp)
{
    for (auto &temp : mp) {
        this->SetEnv(temp.first, temp.second);
    }
    return *this;
}

Situation &Situation::SetClusterType(DeviceType type, int serverCount, int devCount)
{
    devType     = type;
    serverNum = serverCount;
    deviceNum = devCount;
    SetFakeSocVersion(devType);

    return *this;
}

Situation &Situation::SetAcceleratorMode(Accelerator mode)
{
    this->accelerator = mode;
    return *this;
}

bool Situation::IsAicpuAcceleratorMode()
{
    if (this->devType != DeviceType::DEV_TYPE_910_93 && this->devType != DeviceType::DEV_TYPE_910_95) {
        return false;
    }
    // 此处保留HCCL_OP_EXPANSION_MODE兼容库上旧用例，A3继续使用
    return (this->accelerator == Accelerator::AICPU) || (SalGetEnv("HCCL_OP_EXPANSION_MODE") == "AI_CPU");
}

bool Situation::IsCcuAcceleratorMode()
{
    if (this->devType != DeviceType::DEV_TYPE_910_95) {
        return false;
    }
    // 此处保留CCU_ENABLE兼容库上旧用例
    return (this->accelerator == Accelerator::CCU) || (SalGetEnv("CCU_ENABLE") == "2");
}

void Situation::PrintDataVec(void)
{
    auto type = this->dataType;
    for (auto ptr : this->dataVec) {
        switch (type) {
            case HcclDataType::HCCL_DATA_TYPE_FP32:
                std::cout << *(static_cast<float *>(ptr)) << " ";
                break;
            case HcclDataType::HCCL_DATA_TYPE_INT32:
                std::cout << *(static_cast<s32 *>(ptr)) << " ";
                break;
        }
    }
    std::cout << std::endl;
}