#include <fstream>
#include <securec.h>
#include <runnerdb/sim_runner_common.h>
#include "gen_device_topofile.h"
#include "hccl_common_macro.h"
#include "hccl_vm_log.h"
#include "sim_runner_ops.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cstdint>

// 生成topo和ranktable文件需要获取的信息：
/*
 1. EndPointPair硬件链路信息：Port IPAddr/EID; func_id（框内2，出框3）; protocol类型: 框内HCCS，出框ROCE
 2. 获取channel映射表：
   2.1 EID获取IP地址
   2.2 IP地址获取EndPointPair ———— IP之间一一配对
   2.3 插入一条channel
 */

namespace {
struct SimDevEidInfo {
    std::string portId{0};
    std::string name{0};
    std::string ipAddr;
    uint32_t eidIndex{0};
    uint32_t type{0};
    uint32_t dieId{0};
    uint32_t chipId{0};
    uint32_t funcId{0};
};

std::map<uint32_t, std::vector<SimDevEidInfo>> g_devId2EidInfo;
std::map<uint32_t, std::map<std::string, std::pair<uint32_t, uint32_t>>> g_devId2Ip2DieIdAndFuncId;
std::map<uint32_t, uint32_t> g_devId2funcId;
std::map<uint32_t, map<std::string, uint32_t>> g_devId2PortId2DieId;
std::map<uint32_t, map<std::string, uint32_t>> g_devId2PortId2funcId;
std::map<uint32_t, map<std::string, SimDevEidInfo>> g_devId2PortId2EidInfo;
std::map<uint32_t, map<uint32_t, string>> g_uvDevice2Port;

// new
uint64_t g_rdmaHandle = 0x80000000; // 

std::string ipv4_to_128bit_id(const std::string& ip)
{
    unsigned a, b, c, d;

    // 解析 IPv4 字符串
    if (sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4)
        return "";

    // 构造最后 32bit 的 hex
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    // 前 96bit = 24 个 hex 字符全 0
    oss << "000000000000000000000000";

    // IPv4 按大端序逐字节写入
    oss << std::setw(2) << a
        << std::setw(2) << b
        << std::setw(2) << c
        << std::setw(2) << d;

    return oss.str();
}


uint32_t GetRankNumFormTopoMeta(TopoMeta &topoMeta)
{
    uint32_t rankNum = 0;
    for (auto &podMeta : topoMeta) {
        for (auto &serverMeta : podMeta) {
            rankNum += serverMeta.size();
        }
    }
    return rankNum;
}

uint32_t GetServerNumFormTopoMeta(TopoMeta &topoMeta)
{
    uint32_t sererNum = 0;
    for (auto &podMeta : topoMeta) {
        for (auto &serverMeta : podMeta) {
            if (serverMeta.size()) {
                sererNum++;
            }
        }
    }
    return sererNum;
}

void AddEidInfo(uint32_t deviceId, const std::string &port, uint32_t dieId)
{
    SimDevEidInfo eidInfo;
    eidInfo.portId = port;
    eidInfo.dieId  = dieId;
    eidInfo.chipId = deviceId;
    if (g_devId2funcId.count(deviceId) == 0) {
        g_devId2funcId[deviceId] = 1; // todo：区分框内和出框场景
    }
    eidInfo.funcId = g_devId2funcId[deviceId];
    g_devId2funcId[deviceId] += 1;
    g_devId2EidInfo[deviceId].push_back(eidInfo);
    g_devId2PortId2DieId[deviceId][port] = dieId;
    g_devId2PortId2funcId[deviceId][port] = eidInfo.funcId;
    return;
}

uint64_t AddOneDevice(uint64_t serverKey, uint32_t physicalId)
{
    sim::Device device{};
    device.server_id = serverKey;
    device.physical_id = physicalId;
    device.overflow_mode = 0;
    strcpy(device.soc_version, "Ascend950");
    device.status = 0;
    // HCCL_VM_DEBUG("[{}] Add one device serverId {:d}, phyDevId {:d}", __func__, serverKey, physicalId);
    return RunnerDB::Add<sim::Device>(device);
}

uint64_t AddOneServer(uint64_t podId)
{
    sim::Server server;
    server.pod_id = podId;
    return RunnerDB::Add<sim::Server>(server);
}

uint64_t AddOneHost(const std::string &serverIp, uint64_t serverId)
{
    sim::Host host;
    host.server_id = serverId;
    strcpy(host.ip_addr, serverIp.c_str());
    std::cout<<"AddOneHost: "<<serverId<<", "<<serverIp<<std::endl;
    return RunnerDB::Add<sim::Host>(host);
}

uint64_t AddOneRank(uint64_t deviceKey, uint32_t rankId)
{
    sim::Rank rank;
    rank.rank_id = rankId;
    rank.device_id = deviceKey;
    return RunnerDB::Add<sim::Rank>(rank);
}

uint64_t AddOneCcu(uint64_t deviceKey, uint8_t dieId)
{
    sim::Ccu ccu{};
    ccu.device_id = deviceKey;
    ccu.die_id = dieId;
    ccu.status = 0;
    auto ccuId = RunnerDB::Add<sim::Ccu>(ccu);

    sim::CcuResource res{};
    res.ccu_id = ccuId;
    
    auto resId = RunnerDB::Add<sim::CcuResource>(res);
    // HCCL_VM_DEBUG("[{}] Add one ccu: id= {},  key= {}", __func__, ccuId, resId);
    return ccuId;
}

int g_PodIpBase = 192;
int g_InPodServerIpBase = 1;
std::vector<std::pair<int, int>> g_InServreIpPair;
std::vector<std::vector<std::map<std::string, std::string>>> g_port2IpAddr;
}

const static uint32_t MAX_DEV_PER_ROW = 8;
std::vector<std::map<uint32_t, uint64_t>> phyDevId2DevKey;

// todo: 原checker中algName，从哪里获取？
HcclVmResult DeviceTopoGenerator::Init(TopoMeta &topoMeta, const std::vector<std::string>& serverIdx2Ip, const std::string &algName)
{
    HCCL_VM_DEBUG("[{}] Enter topo generator.....", __func__);
    // 初始化topo/ranktable文件名称
    const char *value = std::getenv("TOPO_PATH_NAME");
    if (value != nullptr) {
        std::string prefix(value);
        topoFilePath_ = prefix + "_topo.json";
        rankTableFilePath_ = prefix + "_ranktable.json";
    } else {
        topoFilePath_ = "topo.json";
        rankTableFilePath_ = "ranktable.json";
    }

    // 根据topoMeta获取生成topo.json文件的规格（完整规格topo.json业务可能无法跑通）:预期单server
    if (topoMeta.empty() || topoMeta[0].empty()) {
        HCCL_VM_ERROR("[{}] Wrong topo meta size {}", __func__, topoMeta.size());
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    auto serverNum = topoMeta[0].size();
    device910DXAxisRankNum_.resize(serverNum, 8);
    device910DYAxisRankNum_.resize(serverNum, 8);
    device910DRankSize_.resize(serverNum, 64);
    // for (uint32_t serverIdx = 0; serverIdx < serverNum; serverIdx++) {
    //     uint32_t maxRow = 0;
    //     uint32_t maxCol = 0;
    //     for (const auto &locId : topoMeta[0][serverIdx]) {
    //         auto row = locId / MAX_DEV_PER_ROW;
    //         auto col = locId % MAX_DEV_PER_ROW;
    //         if (maxRow < row) {
    //             maxRow = row;
    //         }
    //         if (maxCol < col) {
    //             maxCol = col;
    //         }
    //     }
    //     device910DXAxisRankNum_[serverIdx] = maxCol + 1;
    //     device910DYAxisRankNum_[serverIdx] = maxRow + 1;
    //     device910DRankSize_[serverIdx] = device910DXAxisRankNum_[serverIdx] * device910DYAxisRankNum_[serverIdx];
    //     HCCL_VM_INFO("[{}] Get topo specifications, serverIdx = {}, max row = {}, max col = {}",
    //         __func__,
    //         serverIdx,
    //         device910DYAxisRankNum_[serverIdx],
    //         device910DXAxisRankNum_[serverIdx]);
    // }

    const char* hcclvmTopoType = std::getenv("HCCLVM_TOPO_TYPE");
    if (hcclvmTopoType == nullptr || std::string(hcclvmTopoType).empty()) { // 普通机型
        HCCL_VM_INFO("[{}] Enter into normal 910D topo.", __func__);

        // 初始化server和host信息 todo: 后续多server涉及runner初始化、host初始化、server初始化
        serverIdx2Id_.clear();
        for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
            for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
                auto serverKey = AddOneServer(superPodIdx);
                auto hostKey = AddOneHost(serverIdx2Ip[serverIdx], serverKey);
                serverIdx2Id_.push_back(serverKey);
                std::cout<<"[InitGenRankTableJson] Add host & server: "<<serverKey<<", "<<hostKey<<", serverIp="<<serverIdx2Ip[serverIdx]<<std::endl;
            }
        }

        g_port2IpAddr.resize(serverNum);
        g_InServreIpPair.resize(serverNum, std::make_pair(10, 10));
        phyDevId2DevKey.resize(serverNum);
        for (uint32_t serverIdx = 0; serverIdx < serverNum; serverIdx++) {
            // 初始化device、ccu和所有port信息 ———— todo: 后续根据芯片类型不同有不同的初始化
            g_port2IpAddr[serverIdx].resize(device910DXAxisRankNum_[serverIdx] * device910DYAxisRankNum_[serverIdx]);
            if (serverNum > 1) {
                // 出框一个server 8张卡
                HCCLVM_CHK_RET(InitDeviceInfoMultiServer(serverIdx));
            } else {
                for (uint32_t row = 0; row < device910DYAxisRankNum_[serverIdx]; row++) {
                    HCCLVM_CHK_RET(InitDeviceInfo(serverIdx, row));
                }
            }

            if (serverNum > 1) {
                // 初始化topo.json文件 ———— 当前认为所有device一样，topo.json只有一份
                HCCLVM_CHK_RET(InitGenTopoJsonMultiServer(serverIdx));
            } else {
                // 初始化topo.json文件 ———— 当前认为所有device一样，topo.json只有一份
                HCCLVM_CHK_RET(InitGenTopoJson(serverIdx));
            }
        }

        // 初始化ranktable.json文件 (todo: algName参数通过环境变量获取)
        HCCLVM_CHK_RET(InitGenRankTableJson(topoMeta, algName));
    } else if (std::string(hcclvmTopoType) == "HF") {
        if (serverNum > 1) {
            HCCL_VM_ERROR("[{}] HF topo not support multi-servers yet.", __func__);
            return HcclVmResult::HCCL_SIM_E_NOT_SUPPORT;
        }
        HCCL_VM_INFO("[{}] Enter into HF 910D topo.", __func__);

        // 初始化server和host信息 todo: 后续多server涉及runner初始化、host初始化、server初始化
        serverIdx2Id_.clear();
        for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
            for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
                auto serverKey = AddOneServer(superPodIdx);
                auto hostKey = AddOneHost(serverIdx2Ip[serverIdx], serverKey);
                serverIdx2Id_.push_back(serverKey);
                std::cout<<"[InitGenRankTableJsonHF] Add host & server: "<<serverKey<<", "<<hostKey<<", serverIp="<<serverIdx2Ip[serverIdx]<<std::endl;
            }
        }


        g_port2IpAddr.resize(serverNum);
        g_InServreIpPair.resize(serverNum, std::make_pair(10, 10));
        phyDevId2DevKey.resize(serverNum);
        for (uint32_t serverIdx = 0; serverIdx < serverNum; serverIdx++) {
            g_port2IpAddr[serverIdx].resize(8 * 2);
            for (uint32_t row = 0; row < 2; row++) {
                HCCLVM_CHK_RET(InitHFDeviceInfo(serverIdx, row));
            }
 
            // 初始化topo.json文件
            HCCLVM_CHK_RET(InitGenTopoJsonHF(serverIdx));
 
            // 初始化ranktable.json文件 (todo: algName参数通过环境变量获取)
            HCCLVM_CHK_RET(InitGenRankTableJsonHF(topoMeta, serverIdx2Ip));
        }
    } else if (std::string(hcclvmTopoType) == "UBX") {
        if (serverNum > 1) {
            HCCL_VM_ERROR("[{}] UBX topo not support multi-servers yet.", __func__);
            return HcclVmResult::HCCL_SIM_E_NOT_SUPPORT;
        }
        HCCL_VM_INFO("[{}] Enter into UBX 910D topo.", __func__);

        // 初始化server和host信息 todo: 后续多server涉及runner初始化、host初始化、server初始化
        serverIdx2Id_.clear();
        for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
            for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
                auto serverKey = AddOneServer(superPodIdx);
                auto hostKey = AddOneHost(serverIdx2Ip[serverIdx], serverKey);
                serverIdx2Id_.push_back(serverKey);
                std::cout<<"[InitGenRankTableJsonUBX] Add host & server: "<<serverKey<<", "<<hostKey<<", serverIp="<<serverIdx2Ip[serverIdx]<<std::endl;
            }
        }

        g_port2IpAddr.resize(serverNum);
        g_InServreIpPair.resize(serverNum, std::make_pair(10, 10));
        phyDevId2DevKey.resize(serverNum);
        g_port2IpAddr[0].resize(4 * 4);
        HCCLVM_CHK_RET(InitUBXDeviceInfo(0));

        // 初始化topo.json文件
        HCCLVM_CHK_RET(InitGenTopoJsonUBX(0));

        // 初始化ranktable.json文件 (todo: algName参数通过环境变量获取)
        HCCLVM_CHK_RET(InitGenRankTableJsonUBX(topoMeta, 0, serverIdx2Ip));
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

/*
 初始化device的port信息
 910D：每个device有18个端口，每个ccu有9个端口。
 die0 ccu: port 0/0 ~ 0/6作为rank内链路，netlayer=0，func_id = 2；
           port 0/7, 0/8作为rank间链路，netlayer=1/2，func_id = 3；
 die1 ccu: port 1/0 ~ 1/6作为rank内链路，netlayer=0，func_id = 2；
           port 1/7, 1/8作为rank间链路，netlayer=1/2，func_id = 3；
*/
HcclVmResult DeviceTopoGenerator::InitDeviceInfo(uint32_t serverIdx, uint32_t rowIdx)
{
    for (uint32_t colSrc = 0; colSrc < device910DXAxisRankNum_[serverIdx]; colSrc++) {
        uint32_t deviceId = rowIdx * device910DXAxisRankNum_[serverIdx] + colSrc;
        auto serverKey = serverIdx2Id_[serverIdx];
        auto deviceKey = AddOneDevice(serverKey, deviceId);
        auto die0Key   = AddOneCcu(deviceKey, 0);
        auto die1Key   = AddOneCcu(deviceKey, 1);
        phyDevId2DevKey[serverIdx][deviceId] = deviceKey;

        // 910D框内：一个ccu有7个port
        for (uint32_t idx = 0; idx < 7; idx++) {
            auto die0Port = "0/" + std::to_string(idx);
            auto die1Port = "1/" + std::to_string(idx);
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, die0Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, die1Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
        }
        // 910D出框：一个ccu有2个port
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, "0/7", ProtocolType::SIM_PROTOCOL_ROCE, 3));
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, "0/8", ProtocolType::SIM_PROTOCOL_ROCE, 3));
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, "1/7", ProtocolType::SIM_PROTOCOL_ROCE, 3));
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, "1/8", ProtocolType::SIM_PROTOCOL_ROCE, 3));
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitDeviceInfoMultiServer(uint32_t serverIdx)
{
    for (uint32_t colSrc = 0; colSrc < device910DXAxisRankNum_[serverIdx]; colSrc++) {
        uint32_t deviceId = colSrc;
        auto serverKey = serverIdx2Id_[serverIdx];
        auto deviceKey = AddOneDevice(serverKey, deviceId);
        auto die0Key   = AddOneCcu(deviceKey, 0);
        auto die1Key   = AddOneCcu(deviceKey, 1);
        phyDevId2DevKey[serverIdx][deviceId] = deviceKey;

        // 910D框内：一个ccu有7个port
        for (uint32_t idx = 0; idx < 7; idx++) {
            auto die0Port = "0/" + std::to_string(idx);
            auto die1Port = "1/" + std::to_string(idx);
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, die0Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, die1Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
        }
        // 910D出框：一个ccu有2个port
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, "0/7", ProtocolType::SIM_PROTOCOL_ROCE, 3));
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, "0/8", ProtocolType::SIM_PROTOCOL_ROCE, 3));
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, "1/7", ProtocolType::SIM_PROTOCOL_ROCE, 3));
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, "1/8", ProtocolType::SIM_PROTOCOL_ROCE, 3));
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitHFDeviceInfo(uint32_t serverIdx, uint32_t rowIdx)
{
    for (uint32_t colSrc = 0; colSrc < 8; colSrc++) {
        uint32_t deviceId = rowIdx * 8 + colSrc;
        auto serverKey = serverIdx2Id_[serverIdx];
        auto deviceKey = AddOneDevice(serverKey, deviceId);
        auto die0Key   = AddOneCcu(deviceKey, 0);
        auto die1Key   = AddOneCcu(deviceKey, 1);
        phyDevId2DevKey[serverIdx][deviceId] = deviceKey;

        // 910D框内：一个ccu有7个port
        for (uint32_t idx = 0; idx < 9; idx++) {
            auto die0Port = "0/" + std::to_string(idx);
            auto die1Port = "1/" + std::to_string(idx);
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, die0Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, die1Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
        }
        // HF出框？
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitUBXDeviceInfo(uint32_t serverIdx)
{
    for (uint32_t colSrc = 0; colSrc < 16; colSrc++) {
        uint32_t deviceId = colSrc;
        auto serverKey = serverIdx2Id_[serverIdx];
        auto deviceKey = AddOneDevice(serverKey, deviceId);
        auto die0Key   = AddOneCcu(deviceKey, 0);
        auto die1Key   = AddOneCcu(deviceKey, 1);
        phyDevId2DevKey[serverIdx][deviceId] = deviceKey;

        // 1. 组内peer2peer port: 0/0, 0/1, 0/2
        for (uint32_t idx = 0; idx < 3; idx++) {
            auto die0Port = "0/" + std::to_string(idx);
            auto die1Port = "1/" + std::to_string(idx);
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die0Key, deviceId, die0Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
            HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, die1Port, ProtocolType::SIM_PROTOCOL_HCCS, 2));
        }
        // 2. 组内peer2net port: 0/4, 0/5, 0/6, 0/7，但其共用一个ip
        std::vector<std::string> peer2NetPorts;
        for (uint32_t idx = 4; idx < 8; idx++) {
            auto die0Port = "0/" + std::to_string(idx);
            peer2NetPorts.push_back(die0Port);
        }
        HCCLVM_CHK_RET(InitSameIpPortsInfo(serverIdx, deviceKey, die0Key, deviceId, peer2NetPorts, ProtocolType::SIM_PROTOCOL_HCCS, 2));
        // 3. net layer1的port: d2h
        HCCLVM_CHK_RET(InitOnePortInfo(serverIdx, deviceKey, die1Key, deviceId, "d2h", ProtocolType::SIM_PROTOCOL_ROCE, 2));
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GetDeviceIdAndCcuId(uint32_t logicDevId, uint8_t dieId, uint64_t &deviceKey, uint64_t &ccuKey)
{
    sim::Device device{};
    HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetDeviceByPhysicalId(logicDevId, device)));
    sim::Ccu ccu{};
    HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetCcuFromDeviceByDieId(device.id, dieId, ccu)));

    deviceKey = device.id;
    ccuKey = ccu.id;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::UpdatePortEidInfo(uint32_t serverIdx, uint32_t phyDevId, const std::string &portName, const std::string &ipAddr)
{
    // 1. 根据portName获取port
    sim::Port port{};
    auto serverKey = serverIdx2Id_[serverIdx];
    HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetPortByName(serverKey, phyDevId, portName, port)));

    auto handle =  g_rdmaHandle++;
    // 2. 更新Port的EID信息
    auto portKey = port.id;
    auto ret = RunnerDB::Update<sim::Port>(portKey, [portKey, handle](sim::Port &p) {
        p.rdma_handle = handle;
        p.status = 1;
    });
    if (!ret) {
        HCCL_VM_ERROR("[{}] update port info failed. ipAddr={}", __func__, ipAddr);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitOnePortInfo(
    uint32_t serverIdx, uint64_t deviceKey, uint64_t ccuKey, uint32_t phyDevId, const std::string &portName, ProtocolType protocolType, uint32_t funcId)
{
    std:string ipAddr = std::to_string(g_PodIpBase) + "." + std::to_string(g_InPodServerIpBase + serverIdx) + ".";
    if (g_InServreIpPair[serverIdx].second + 1 > 100) {
        g_InServreIpPair[serverIdx].first += 1;
        if (g_InServreIpPair[serverIdx].first > 100) { // a.b的取值范围{10.10  ~ 100.100}
            HCCL_VM_ERROR("[{}] Create port ip addr failed. {:s}{:d}.{:d}", __func__, ipAddr, g_InServreIpPair[serverIdx].first, g_InServreIpPair[serverIdx].second);
            return HcclVmResult::HCCL_SIM_E_NOT_SUPPORT;
        }
        g_InServreIpPair[serverIdx].second = 10;
    }
    ipAddr += std::to_string(g_InServreIpPair[serverIdx].first) + "." + std::to_string(g_InServreIpPair[serverIdx].second++);

    g_port2IpAddr[serverIdx][phyDevId][portName] = ipAddr;

    sim::Port uPortInfo{};
    uPortInfo.device_id = deviceKey;
    uPortInfo.ccu_id = ccuKey;
    uPortInfo.func_id = funcId;
    strcpy(uPortInfo.name, portName.c_str());
    strcpy(uPortInfo.ip_addr, ipAddr.c_str());
    uPortInfo.protocol = protocolType;
    auto uPortId = RunnerDB::Add<sim::Port>(uPortInfo);
    
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitSameIpPortsInfo(
    uint32_t serverIdx, uint64_t deviceKey, uint64_t ccuKey, uint32_t phyDevId, const std::vector<std::string> &portNames, ProtocolType protocolType, uint32_t funcId)
{
    std:string ipAddr = std::to_string(g_PodIpBase) + "." + std::to_string(g_InPodServerIpBase + serverIdx) + ".";
    if (g_InServreIpPair[serverIdx].second + 1 > 100) {
        g_InServreIpPair[serverIdx].first += 1;
        if (g_InServreIpPair[serverIdx].first > 100) { // a.b的取值范围{10.10  ~ 100.100}
            HCCL_VM_ERROR("[{}] Create port ip addr failed. {:s}{:d}.{:d}", __func__, ipAddr, g_InServreIpPair[serverIdx].first, g_InServreIpPair[serverIdx].second);
            return HcclVmResult::HCCL_SIM_E_NOT_SUPPORT;
        }
        g_InServreIpPair[serverIdx].second = 10;
    }
    ipAddr += std::to_string(g_InServreIpPair[serverIdx].first) + "." + std::to_string(g_InServreIpPair[serverIdx].second++);

    for (const auto &portName : portNames) {
        g_port2IpAddr[serverIdx][phyDevId][portName] = ipAddr;

        sim::Port uPortInfo{};
        uPortInfo.device_id = deviceKey;
        uPortInfo.ccu_id = ccuKey;
        uPortInfo.func_id = funcId;
        strcpy(uPortInfo.name, portName.c_str());
        strcpy(uPortInfo.ip_addr, ipAddr.c_str());
        uPortInfo.protocol = protocolType;
        auto uPortId = RunnerDB::Add<sim::Port>(uPortInfo);
    }
    
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitEndPointPairInfo(uint32_t serverIdx, uint32_t srcPhyDevId, uint32_t dstPhyDevId, const std::string &srcPortName, const std::string &dstPortName, bool isInServer)
{
    // 1. 根据portName获取port
    sim::Port srcPort{};
    auto serverKey = serverIdx2Id_[serverIdx];
    HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetPortByName(serverKey, srcPhyDevId, srcPortName, srcPort)));

    sim::Port dstPort{};
    if (isInServer) {
        // 出框场景，只有srcPort，没有dstPort
        HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetPortByName(serverKey, dstPhyDevId, dstPortName, dstPort)));
    }

    // 2. 新增EndPointPair
    sim::EndPointPair epPair{};
    epPair.src_port = srcPort.id;
    epPair.dst_port = dstPort.id;
    HCCL_VM_DEBUG("[{}] srcPort = {}, {} dstPort= {}, {}", __func__, srcPortName, epPair.src_port, dstPortName, epPair.dst_port);
    auto epPairId = RunnerDB::Add<sim::EndPointPair>(epPair);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenXNetlayer0TopoLink(uint32_t serverIdx, uint32_t rowIdx, json &edge_list)
{
    vector<uint32_t> recorder(device910DXAxisRankNum_[serverIdx], 0);
    uint32_t linkCnt = 0;
    for (uint32_t colSrc = 0; colSrc < device910DXAxisRankNum_[serverIdx]; colSrc++) {
        for (uint32_t colDst = colSrc + 1; colDst < device910DXAxisRankNum_[serverIdx]; colDst++) {
            linkCnt += 1;
            std::string u_port = "0/";
            std::string v_port = "0/";

            u_port += std::to_string(recorder[colSrc]++);
            v_port += std::to_string(recorder[colDst]++);

            uint32_t uDeviceId = rowIdx * device910DXAxisRankNum_[serverIdx] + colSrc;
            uint32_t vDeviceId = rowIdx * device910DXAxisRankNum_[serverIdx] + colDst;

            edge_list.push_back(json{{"net_layer", 0},
                {"link_type", "PEER2PEER"},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_port}},
                {"local_b", vDeviceId},
                {"local_b_ports", {v_port}},
                {"position", "DEVICE"},
                {"topo_type", "1DMESH"}});
            
            // 保存EndPointPair信息
            HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, vDeviceId, u_port, v_port, true));
            
            AddEidInfo(uDeviceId, u_port, 0);
            AddEidInfo(vDeviceId, v_port, 0);
            g_uvDevice2Port[uDeviceId][vDeviceId] = u_port;
            g_uvDevice2Port[vDeviceId][uDeviceId] = v_port;
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenYNetlayer0TopoLink(uint32_t serverIdx, uint32_t colIdx, json &edge_list)
{
    uint32_t linkCnt = 0;
    vector<uint32_t> recorder(device910DYAxisRankNum_[serverIdx], 0);
    for (uint32_t rowSrc = 0; rowSrc < device910DYAxisRankNum_[serverIdx]; rowSrc++) {
        for (uint32_t rowDst = rowSrc + 1; rowDst < device910DYAxisRankNum_[serverIdx]; rowDst++) {
            linkCnt += 1;
            std::string u_port = "1/";
            std::string v_port = "1/";

            u_port += std::to_string(recorder[rowSrc]++);
            v_port += std::to_string(recorder[rowDst]++);
            uint32_t uDeviceId = rowSrc * device910DYAxisRankNum_[serverIdx] + colIdx;
            uint32_t vDeviceId = rowDst * device910DYAxisRankNum_[serverIdx] + colIdx;

            edge_list.push_back(json{{"net_layer", 0},
                {"link_type", "PEER2PEER"},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_port}},
                {"local_b", vDeviceId},
                {"local_b_ports", {v_port}},
                {"position", "DEVICE"},
                {"topo_type", "1DMESH"}});

            // 保存EndPointPair信息
            HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, vDeviceId, u_port, v_port, true));

            AddEidInfo(uDeviceId, u_port, 1);
            AddEidInfo(vDeviceId, v_port, 1);
            g_uvDevice2Port[uDeviceId][vDeviceId] = u_port;
            g_uvDevice2Port[vDeviceId][uDeviceId] = v_port;
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenNetlayer1TopoLink(uint32_t serverIdx, uint32_t row, json &edge_list)
{
    for (uint32_t col = 0; col < device910DYAxisRankNum_[serverIdx]; col++) {
        uint32_t uDeviceId = row * device910DYAxisRankNum_[serverIdx] + col;
        std::string u_die0_port = "0/7";
        edge_list.push_back(json{{"net_layer", 1},
            {"link_type", "PEER2NET"},
            {"protocols", {"UB_CTP"}},
            {"local_a", uDeviceId},
            {"local_a_ports", {u_die0_port}},
            {"position", "DEVICE"},
            {"topo_type", "CLOS"}});
        
        std::string u_die1_port = "1/7";
        edge_list.push_back(json{{"net_layer", 1},
            {"link_type", "PEER2NET"},
            {"protocols", {"UB_CTP"}},
            {"local_a", uDeviceId},
            {"local_a_ports", {u_die1_port}},
            {"position", "DEVICE"},
            {"topo_type", "CLOS"}});
        
        // 保存EndPointPair信息
        HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, -1, u_die0_port, {}, false));
        HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, -1, u_die1_port, {}, false));

        AddEidInfo(uDeviceId, u_die0_port, 0);
        AddEidInfo(uDeviceId, u_die1_port, 0);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenNetlayer2TopoLink(uint32_t serverIdx, uint32_t row, json &edge_list)
{
    for (uint32_t col = 0; col < device910DYAxisRankNum_[serverIdx]; col++) {
        uint32_t uDeviceId = row * device910DYAxisRankNum_[serverIdx] + col;
        std::string u_port = "0/8";
        edge_list.push_back(json{{"net_layer", 2},
            {"link_type", "PEER2NET"},
            {"protocols", {"UB_CTP"}},
            {"local_a", uDeviceId},
            {"local_a_ports", {u_port}},
            {"position", "HOST"},
            {"topo_type", "CLOS"}});
        
        // 保存EndPointPair信息
        HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, -1, u_port, {}, false));

        AddEidInfo(uDeviceId, u_port, 1);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 每行8个芯片
const static uint32_t TOPO_HF_DEV_NUM_PER_ROW = 8;
const static uint32_t DEV_NUM_PER_SERVER = 16;
HcclVmResult DeviceTopoGenerator::InitGenTopoJsonHF(uint32_t serverIdx)
{
    uint32_t rankSizeNum = DEV_NUM_PER_SERVER;  // 需要更大ranksize时，需要修改该值 = 64，最大支持64

    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "Atlas 550";
    topoJson["peer_count"] = rankSizeNum;

    // 创建 peer_list 数组
    json peer_list = json::array();
    for (uint32_t id = 0; id < rankSizeNum; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 生成 edge_list
    json edge_list = json::array();

    // 生成行方向，即die1上的链路
    uint32_t linkCnt = 0;
    for (uint32_t row = 0; row < 2; row++) { // 当前一个server只有两行
        vector<uint32_t> recorder(TOPO_HF_DEV_NUM_PER_ROW, 0);
        linkCnt = 0;
        for (uint32_t colSrc = 0; colSrc < TOPO_HF_DEV_NUM_PER_ROW; colSrc++) {
            for (uint32_t colDst = colSrc + 1; colDst < TOPO_HF_DEV_NUM_PER_ROW; colDst++) {
                linkCnt += 1;
                std::string u_addr = "1/";
                std::string v_addr = "1/";

                u_addr += std::to_string(recorder[colSrc]);
                v_addr += std::to_string(TOPO_HF_DEV_NUM_PER_ROW - recorder[colSrc]);
                recorder[colSrc]++;
                uint32_t uDeviceId = row * TOPO_HF_DEV_NUM_PER_ROW + colSrc;
                uint32_t vDeviceId = row * TOPO_HF_DEV_NUM_PER_ROW + colDst;

                edge_list.push_back(json{{"net_layer", 0},
                    {"link_type", "PEER2PEER"},
                    {"topo_type", "1DMESH"},
                    {"topo_instance_id", 0},
                    {"protocols", {"UB_CTP"}},
                    {"local_a", uDeviceId},
                    {"local_a_ports", {u_addr}},
                    {"local_b", vDeviceId},
                    {"local_b_ports", {v_addr}},
                    {"position", "DEVICE"}});
                
                // 保存EndPointPair信息
                HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, vDeviceId, u_addr, v_addr, true));

                AddEidInfo(uDeviceId, u_addr, 0);
                AddEidInfo(vDeviceId, v_addr, 0);
                g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
                g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
            }
        }
    }

    // 生成列方向，即die0上的链路
    vector<uint32_t> recorder1(TOPO_HF_DEV_NUM_PER_ROW, 0);
    vector<uint32_t> recorder2(TOPO_HF_DEV_NUM_PER_ROW, 0);
    for (uint32_t colSrc = 0; colSrc < TOPO_HF_DEV_NUM_PER_ROW; colSrc++) {
        for (uint32_t colDst = 0; colDst < TOPO_HF_DEV_NUM_PER_ROW; colDst++) {
            std::string u_addr = "0/";
            std::string v_addr = "0/";

            u_addr += std::to_string(recorder1[colSrc]++);
            v_addr += std::to_string(recorder2[colDst]++);
            uint32_t uDeviceId = colSrc;
            uint32_t vDeviceId = colDst + TOPO_HF_DEV_NUM_PER_ROW;

            edge_list.push_back(json{{"net_layer", 0},
                {"link_type", "PEER2PEER"},
                {"topo_type", "1DMESH"},
                {"topo_instance_id", 0},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_addr}},
                {"local_b", vDeviceId},
                {"local_b_ports", {v_addr}},
                {"position", "DEVICE"}});
            
            // 保存EndPointPair信息
            HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, vDeviceId, u_addr, v_addr, true));

            AddEidInfo(uDeviceId, u_addr, 1);
            AddEidInfo(vDeviceId, v_addr, 1);
            g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
            g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
        }
    }

    topoJson["edge_count"] = edge_list.size();
    topoJson["edge_list"] = edge_list;

    std::ofstream topoFile(topoFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, topoFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << topoJson.dump(4);
    topoFile.close();
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 每行8个芯片
const static uint32_t UBX_GROUP_SIZE = 4;
const static uint32_t UBX_DEV_NUM_PER_SERVER = 16;
HcclVmResult DeviceTopoGenerator::InitGenTopoJsonUBX(uint32_t serverIdx)
{
    const uint32_t rankSizeNum = UBX_DEV_NUM_PER_SERVER;  // 固定16个rank
    const uint32_t groupSize   = UBX_GROUP_SIZE;     // 每组4个rank（0-3,4-7,8-11,12-15）
    const uint32_t groupCount  = rankSizeNum / groupSize;  // 共4组

    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "UBX";  // 改为UBX硬件类型
    topoJson["peer_count"] = rankSizeNum;

    // 1. 生成peer_list数组（所有rank节点）
    json peer_list = json::array();
    for (uint32_t id = 0; id < rankSizeNum; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 2. 生成edge_list（链路列表）
    json edge_list = json::array();

    // 2.1 生成组内全连接链路（0-3,4-7,8-11,12-15每组内全连接）
    // 端口使用1/0,1/1,1/2循环分配
    for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
        uint32_t groupStart = groupIdx * groupSize;  // 每组起始rank ID
        uint32_t portIdx = 0;                        // 端口索引（0,1,2循环）
        
        // 1DMESH网络
        vector<uint32_t> recorder(groupSize, 0);
        for (uint32_t srcInGroup = 0; srcInGroup < groupSize; ++srcInGroup) {
            for (uint32_t dstInGroup = srcInGroup + 1; dstInGroup < groupSize; ++dstInGroup) {
                uint32_t uDeviceId = groupStart + srcInGroup;
                uint32_t vDeviceId = groupStart + dstInGroup;
                
                // 分配端口（1/0,1/1,1/2循环）
                std::string u_addr = "0/" + std::to_string(recorder[srcInGroup]++);
                std::string v_addr = "0/" + std::to_string(recorder[dstInGroup]++);
                portIdx++;

                // 添加组内链路信息
                edge_list.push_back(json{
                    {"net_layer", 0},
                    {"link_type", "PEER2PEER"},
                    {"topo_type", "1DMESH"},
                    {"topo_instance_id", groupIdx},
                    {"protocols", {"UB_CTP", "UB_MEM"}},
                    {"local_a", uDeviceId},
                    {"local_a_ports", {u_addr}},
                    {"local_b", vDeviceId},
                    {"local_b_ports", {v_addr}},
                    {"position", "DEVICE"}
                });

                // 保存EndPointPair信息
                HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, uDeviceId, vDeviceId, u_addr, v_addr, true));

                // 记录Eid信息和端口映射
                AddEidInfo(uDeviceId, u_addr, 0);
                AddEidInfo(vDeviceId, v_addr, 0);
                g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
                g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
            }
        }
    }

    // 2.2 生成每个rank到netlayer1节点的链路（1/4,1/5,1/6,1/7端口）
    // 假设netlayer1的节点ID为16-19（可根据实际情况调整）
    const uint32_t netLayer1StartId = 16;  // netlayer1节点起始ID
    const uint32_t netLayer1Count = 4;     // 4个netlayer1节点
    for (uint32_t rankId = 0; rankId < rankSizeNum; ++rankId) {
        json localPorts = json::array();
        for (uint32_t nl1Idx = 0; nl1Idx < netLayer1Count; ++nl1Idx) {
            // 分配端口：1/4,1/5,1/6,1/7
            std::string rankPort = "0/" + std::to_string(4 + nl1Idx);
            localPorts.push_back(rankPort);
            // 保存EndPointPair信息
            HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, rankId, -1, rankPort, {}, false));
        }
        // 添加到netlayer1节点的链路
        edge_list.push_back(json{
            {"net_layer", 0},  // 注意：net_layer设为0
            {"link_type", "PEER2NET"},
            {"topo_type", "CLOS"},
            {"topo_instance_id", 4},
            {"protocols", {"UB_CTP", "UB_MEM"}},
            {"local_a", rankId},
            {"local_a_ports", localPorts},
            {"position", "DEVICE"}
        });

        // d2h
        auto d2hPort = "d2h";
        HCCLVM_CHK_RET(InitEndPointPairInfo(serverIdx, rankId, -1, d2hPort, {}, false));
        // 添加到netlayer1节点的链路
        edge_list.push_back(json{
            {"net_layer", 1},  // 注意：net_layer设为1
            {"link_type", "PEER2NET"},
            {"topo_type", "CLOS"},
            {"topo_instance_id", 4},
            // {"protocols", {"RDMA"}},
            {"protocols", {"UB_CTP", "UB_MEM"}},
            {"local_a", rankId},
            {"local_a_ports", {d2hPort}},
            {"position", "HOST"}
        });
    }

    // 3. 填充edge相关信息
    topoJson["edge_count"] = edge_list.size();
    topoJson["edge_list"] = edge_list;

    std::ofstream topoFile(topoFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, topoFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << topoJson.dump(4);
    topoFile.close();
    std::cout << "Generated UBX topo file: " << topoFilePath_ << std::endl;

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitGenTopoJson(uint32_t serverIdx)
{
    // todo: A2/A3的拓扑数据
    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "910D-2D-Fullmsh_64_plus_1";
    topoJson["peer_count"] = device910DRankSize_[serverIdx];

    // 创建 peer_list 数组
    json peer_list = json::array();
    for (uint32_t id = 0; id < device910DRankSize_[serverIdx]; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 生成 edge_list
    json edge_list = json::array();

    /* ======== 生成netlayer0层级链路： X + Y轴方向 ========= */
    // X轴（Die0）
    uint32_t linkCnt = 0;
    for (uint32_t row = 0; row < device910DYAxisRankNum_[serverIdx]; row++) {
        HCCLVM_CHK_RET(GenXNetlayer0TopoLink(serverIdx, row, edge_list));
    }
    // Y轴（Die1）
    for (uint32_t col = 0; col < device910DXAxisRankNum_[serverIdx]; col++) {
        HCCLVM_CHK_RET(GenYNetlayer0TopoLink(serverIdx, col, edge_list));
    }

    /* ======== 生成netlayer1 && netlayer2层级链路 ========= */
    for (uint32_t row = 0; row < device910DXAxisRankNum_[serverIdx]; row++) {
        HCCLVM_CHK_RET(GenNetlayer1TopoLink(serverIdx, row, edge_list));
    }
    for (uint32_t row = 0; row < device910DXAxisRankNum_[serverIdx]; row++) {
        HCCLVM_CHK_RET(GenNetlayer2TopoLink(serverIdx, row, edge_list));
    }
    /* ======== 生成netlayer1 && netlayer2层级链路 ========= */

    topoJson["edge_count"] = edge_list.size();
    topoJson["edge_list"] = edge_list;

    std::ofstream topoFile(topoFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, topoFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << topoJson.dump(4);
    topoFile.close();

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

const uint32_t RANK_NUM_PER_SERVER_NHR = 8; // 出框算法，一个server有8个卡
HcclVmResult DeviceTopoGenerator::InitGenTopoJsonMultiServer(uint32_t serverIdx)
{
    // todo: A2/A3的拓扑数据
    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "910D-2D-Fullmsh_64_plus_1";
    topoJson["peer_count"] = RANK_NUM_PER_SERVER_NHR;

    // 创建 peer_list 数组
    json peer_list = json::array();
    for (uint32_t id = 0; id < RANK_NUM_PER_SERVER_NHR; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 生成 edge_list
    json edge_list = json::array();

    /* ======== 生成netlayer0层级链路： X + Y轴方向 ========= */
    // X轴（Die0）
    uint32_t linkCnt = 0;
    HCCLVM_CHK_RET(GenXNetlayer0TopoLink(serverIdx, 0, edge_list));

    /* ======== 生成netlayer1 && netlayer2层级链路 ========= */
    HCCLVM_CHK_RET(GenNetlayer1TopoLink(serverIdx, 0, edge_list));
    /* ======== 生成netlayer1 && netlayer2层级链路 ========= */

    topoJson["edge_count"] = edge_list.size();
    topoJson["edge_list"] = edge_list;

    std::ofstream topoFile(topoFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, topoFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << topoJson.dump(4);
    topoFile.close();

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::string to_hex(uint64_t v)
{
    std::ostringstream oss;
    oss << std::hex
        << std::setw(16)
        << std::setfill('0')
        << v;
    return oss.str();
}

HcclVmResult DeviceTopoGenerator::GenRankNetLayer0Node(
    TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level)
{
    json addr_list = json::array();
    json addr;
    level["net_layer"] = 0;
    string tmpNetIns = "az0-rack" + std::to_string(serverIdx);
    tmpNetIns += std::to_string(superPodIdx);
    tmpNetIns += std::to_string(serverIdx);
    level["net_instance_id"] = tmpNetIns;
    level["net_type"] = "TOPO_FILE_DESC";
    level["net_addr"] = "";
    uint32_t uRankId = topoMeta[superPodIdx][serverIdx][rankIdx];

    for (uint32_t vLocalId = 0; vLocalId < rankNum; vLocalId++) {
        if (rankIdx == vLocalId) {
            continue;
        }
        uint32_t vRankId = topoMeta[superPodIdx][serverIdx][vLocalId];
        if ((uRankId / 8 != vRankId / 8) && (uRankId % 8 != vRankId % 8)) {
            continue;
        }
        addr["addr_type"] = "EID";
        string strPorts = g_uvDevice2Port[uRankId % 64][vRankId % 64];
        auto ipAddr = g_port2IpAddr[serverIdx][uRankId].at(strPorts);
        auto eidStr = ipv4_to_128bit_id(ipAddr);
        addr["addr"] = eidStr;
        HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uRankId % 64, strPorts, ipAddr));

        addr["ports"] = {strPorts};
        addr["plane_id"] = "planeA";
        addr_list.push_back(addr);
        
        g_devId2Ip2DieIdAndFuncId[uRankId][ipAddr] = std::pair<uint32_t, uint32_t>(
            g_devId2PortId2DieId[uRankId][strPorts], g_devId2PortId2funcId[uRankId][strPorts]);

        for (uint32_t k = 0; k < g_devId2EidInfo[uRankId].size(); k++) {
            if (g_devId2EidInfo[uRankId][k].portId == strPorts) {
                g_devId2EidInfo[uRankId][k].ipAddr = ipAddr;
            }
        }
    }
    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenRankNetLayer0NodeTest(
    TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t localId, uint32_t rankNum, json &level)
{
    level["net_layer"] = 0;
    level["net_instance_id"] = "az0-rack" + std::to_string(serverIdx);
    level["net_type"] = "TOPO_FILE_DESC";
    level["net_addr"] = "";
    uint32_t rankId = topoMeta[superPodIdx][serverIdx][localId];

    json addr_list = json::array();
    json addr;
    addr["addr_type"] = "EID";
    auto strPort = "0/0";
    auto ipAddr = g_port2IpAddr[serverIdx][rankId].at(strPort);
    auto eidStr = ipv4_to_128bit_id(ipAddr);
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, rankId % 64, strPort, ipAddr));

    addr["addr"] = eidStr;
    addr["ports"] = {strPort};
    addr["plane_id"] = "planeA";
    addr_list.push_back(addr);

    g_devId2Ip2DieIdAndFuncId[rankId][ipAddr] = std::pair<uint32_t, uint32_t>(
        g_devId2PortId2DieId[rankId][strPort], g_devId2PortId2funcId[rankId][strPort]);

    for (uint32_t k = 0; k < g_devId2EidInfo[rankId].size(); k++) {
        if (g_devId2EidInfo[rankId][k].portId == strPort) {
            g_devId2EidInfo[rankId][k].ipAddr = ipAddr;
        }
    }
    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenRankNetLayer1Node(
    TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t localId, uint32_t rankNum, json &level)
{
    level["net_layer"] = 1;
    level["net_instance_id"] = "az0";
    level["net_type"] = "CLOS";
    level["net_addr"] = "";
    uint32_t rankId = topoMeta[superPodIdx][serverIdx][localId];

    json addr_list = json::array();
    json addrDie0;
    addrDie0["addr_type"] = "EID";
    auto strPortDie0 = "0/7";
    auto ipAddrDie0 = g_port2IpAddr[serverIdx][rankId].at(strPortDie0);
    auto eidStrDie0 = ipv4_to_128bit_id(ipAddrDie0);
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, rankId % 64, strPortDie0, ipAddrDie0));

    addrDie0["addr"] = eidStrDie0;
    addrDie0["ports"] = {strPortDie0};
    addrDie0["plane_id"] = "plane0";

    json addrDie1;
    addrDie1["addr_type"] = "EID";
    auto strPortDie1 = "1/7";
    auto ipAddrDie1 = g_port2IpAddr[serverIdx][rankId].at(strPortDie1);
    auto eidStrDie1 = ipv4_to_128bit_id(ipAddrDie1);
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, rankId % 64, strPortDie1, ipAddrDie1));

    addrDie1["addr"] = eidStrDie1;
    addrDie1["ports"] = {strPortDie1};
    addrDie1["plane_id"] = "plane1";
    addr_list.push_back(addrDie0);
    addr_list.push_back(addrDie1);

    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenServerRanktable(TopoMeta &topoMeta, uint32_t &rankCnt, uint32_t superPodIdx, uint32_t serverIdx, const std::string &algName, json &rankList)
{
    uint32_t logicDevCnt = 0;
    for (uint32_t rankIdx = 0; rankIdx < topoMeta[superPodIdx][serverIdx].size(); ++rankIdx) {
        json rank;
        auto rankId = rankCnt++;
        auto logicDevId = logicDevCnt++;
        rank["rank_id"] = rankId;
        auto phyDevId = topoMeta[superPodIdx][serverIdx][rankIdx];
        rank["device_id"] = phyDevId;
        // todo: 后续如果要支持多个超节点，这边要如何处理呢

        auto devKey = phyDevId2DevKey[serverIdx].at(phyDevId);
        AddOneRank(devKey, rankId);

        rank["local_id"] = phyDevId;
        auto serverKey = serverIdx2Id_[serverIdx];
        if (sim::UpdateDeviceLogicId(serverKey, phyDevId, logicDevId) != ACL_SUCCESS) {
            HCCL_VM_ERROR("[{}] get device by logic id 0 failed", __func__);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        auto rankNum = topoMeta[superPodIdx][serverIdx].size();
        json levelList = json::array();
        json level0;
        if (GenRankNetLayer0Node(topoMeta, superPodIdx, serverIdx, rankIdx, rankNum, level0) !=
            HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("[{}] Failed to gen netlayer0 node! rankid: {}", __func__, phyDevId);
            continue;
        }
        levelList.push_back(level0);

        json level1;
        if (GenRankNetLayer1Node(topoMeta, superPodIdx, serverIdx, rankIdx, rankNum, level1) !=
            HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("[{}] Failed to gen netlayer1 node! rankid: {}", __func__, phyDevId);
            continue;
        }
        levelList.push_back(level1);

        rank["level_list"] = levelList;
        rankList.push_back(rank);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenRankNetHFNode(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level)
{
    json addr_list = json::array();
    json addr;
    level["net_layer"] = 0;
    string tmpNetIns = "az0-rack";
    tmpNetIns += std::to_string(superPodIdx);
    tmpNetIns += std::to_string(serverIdx);
    level["net_instance_id"] = tmpNetIns;
    level["net_type"] = "TOPO_FILE_DESC";
    level["net_addr"] = "";
    uint32_t uRankId = topoMeta[superPodIdx][serverIdx][rankIdx];

    for (uint32_t vLocalId = 0; vLocalId < rankNum; vLocalId++) {
        if (rankIdx == vLocalId) {
            continue;
        }
        uint32_t vRankId = topoMeta[superPodIdx][serverIdx][vLocalId];
        addr["addr_type"] = "EID";

        auto uPhyDevId = uRankId % DEV_NUM_PER_SERVER;
        auto vPhyDevId = vRankId % DEV_NUM_PER_SERVER;

        string strPorts = g_uvDevice2Port[uPhyDevId][vRankId];
        auto ipAddr = g_port2IpAddr[serverIdx][uRankId].at(strPorts);
        auto eidStr = ipv4_to_128bit_id(ipAddr);
        HCCL_VM_DEBUG("[{}] Gen HF EID {}-from ip-{}", __func__, eidStr.c_str(), ipAddr.c_str());
        HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, strPorts, ipAddr));

        addr["addr"] = eidStr;
        addr["ports"] = {strPorts};
        if (vLocalId / 8 == rankIdx / 8) {
            addr["plane_id"] = "plane0";
        } else {
            addr["plane_id"] = "plane1";
        }
        
        addr_list.push_back(addr);
        g_devId2Ip2DieIdAndFuncId[uRankId][ipAddr] = std::pair<uint32_t, uint32_t>(
            g_devId2PortId2DieId[uRankId][strPorts], g_devId2PortId2funcId[uRankId][strPorts]);

        for (uint32_t k = 0; k < g_devId2EidInfo[uRankId].size(); k++) {
            if (g_devId2EidInfo[uRankId][k].portId == strPorts) {
                g_devId2EidInfo[uRankId][k].ipAddr = ipAddr;
            }
        }
    }
    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitGenRankTableJsonHF(TopoMeta& topoMeta, const std::vector<std::string>& serverIdx2Ip)
{
    uint32_t serverNum = GetServerNumFormTopoMeta(topoMeta);
    uint32_t rankNum = GetRankNumFormTopoMeta(topoMeta);

    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    uint32_t rankId = 0;
    json rankList = json::array();
    for (uint32_t i = 0; i < topoMeta.size(); ++i) {
        for (uint32_t j = 0; j < topoMeta[i].size(); ++j) {
            for (uint32_t k = 0; k < topoMeta[i][j].size(); ++k) {
                json rank;
                auto logicDevId = rankId++;
                rank["rank_id"] = logicDevId;
                // 后续如果要支持多个超节点，这边要如何处理呢
                auto phyDevId = topoMeta[i][j][k];
                rank["device_id"] = phyDevId;
                rank["local_id"] = phyDevId;

                auto devKey = phyDevId2DevKey[j].at(phyDevId);
                AddOneRank(devKey, logicDevId);
                auto serverKey = serverIdx2Id_[j];
                if (sim::UpdateDeviceLogicId(serverKey, phyDevId, logicDevId) != ACL_SUCCESS) {
                    HCCL_VM_ERROR("[{}] get device by logic id 0 failed", __func__);
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }

                json levelList = json::array();
                json level0;
                if (GenRankNetHFNode(topoMeta, i, j, k, topoMeta[i][j].size(), level0) !=
                    HcclVmResult::HCCL_SIM_SUCCESS) {
                    HCCL_VM_ERROR("Failed to gen netlayer0 node! rankid: {}", topoMeta[i][j][k]);
                    continue;
                }
                levelList.push_back(level0);
                rank["level_list"] = levelList;
                rankList.push_back(rank);
            }
        }
    }
    rankTableJson["rank_list"] = rankList;

    std::ofstream topoFile(rankTableFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, rankTableFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << rankTableJson.dump(4);
    topoFile.close();

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 生成netlayer0（组内全连接）的rank信息
HcclVmResult DeviceTopoGenerator::GenRankNetLayer0HFNode(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level)
{
    json addr_list = json::array();
    json addr;
    
    // 基础字段赋值（匹配目标格式）
    level["net_layer"] = 0;
    // 生成net_instance_id：az0-rack0-podX（X为组索引）
    uint32_t groupIdx = topoMeta[superPodIdx][serverIdx][rankIdx] / 4; // 0-3→pod0,4-7→pod1...
    std::string tmpNetIns = "az0-rack0-pod" + std::to_string(groupIdx);
    level["net_instance_id"] = tmpNetIns;
    level["net_type"] = "TOPO_FILE_DESC";
    level["net_attr"] = ""; // 匹配目标格式的空字段
    
    uint32_t uRankId = topoMeta[superPodIdx][serverIdx][rankIdx];
    uint32_t groupStart = groupIdx * 4;
    uint32_t groupEnd = groupStart + 3;

    // 遍历同组内的其他rank（组内全连接）
    for (uint32_t vRankId = groupStart; vRankId <= groupEnd; vRankId++) {
        if (uRankId == vRankId) continue;

        auto uPhyDevId = uRankId % 16;
        auto vPhyDevId = vRankId % 16;

        // 获取端口信息
        std::string strPorts = g_uvDevice2Port[uPhyDevId][vPhyDevId];
        addr["ports"] = {strPorts};

        addr["addr_type"] = "EID";
        // 生成IP地址（按UBX拓扑规则，示例格式：223.0.X.Y）
        auto ipAddr = g_port2IpAddr[serverIdx][uRankId].at(strPorts);
        auto eidStr = ipv4_to_128bit_id(ipAddr);
        HCCL_VM_DEBUG("[{}] Gen HF EID {}-from ip-{}", __func__, eidStr.c_str(), ipAddr.c_str());
        HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, strPorts, ipAddr));
        
        // 不设置plane_id（netlayer0不需要）
        addr_list.push_back(addr);

        // 填充全局映射表
        g_devId2Ip2DieIdAndFuncId[uRankId][ipAddr] = 
            std::pair<uint32_t, uint32_t>(
                g_devId2PortId2DieId[uRankId][strPorts], 
                g_devId2PortId2funcId[uRankId][strPorts]
            );

        // 更新EidInfo的IP地址
        for (uint32_t k = 0; k < g_devId2EidInfo[uRankId].size(); k++) {
            if (g_devId2EidInfo[uRankId][k].portId == strPorts) {
                g_devId2EidInfo[uRankId][k].ipAddr = ipAddr;
            }
        }
    }

    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 生成netlayer1（CLOS网络）的rank信息
HcclVmResult DeviceTopoGenerator::GenRankNetLayer1HFNode(uint32_t uRankId, json &level)
{
    json addr_list = json::array();
    json addr;
    
    // 基础字段赋值（匹配目标格式）
    level["net_layer"] = 1;
    level["net_instance_id"] = "az0";
    level["net_type"] = "CLOS";
    level["net_attr"] = "";

    // 4个netlayer1节点的配置（端口1/4,1/5,1/6,1/7）
    const std::vector<std::string> planeIds = {"plane0", "plane1", "plane2", "plane3"};
    const uint32_t netLayer1StartId = 16;
    uint32_t groupIdx = uRankId / 4; // 所属组索引
    
    for (uint32_t nl1Idx = 0; nl1Idx < 4; nl1Idx++) {
        uint32_t nl1Id = netLayer1StartId + nl1Idx;

        // 获取端口信息（1/4,1/5,1/6,1/7）
        std::string strPorts = g_uvDevice2Port[uRankId][nl1Id];
        addr["ports"] = {strPorts};
        addr["plane_id"] = planeIds[nl1Idx]; // 设置plane_id

        addr["addr_type"] = "EID";
        // 生成IP地址（匹配示例格式：223.0.X.Y）
        std::string baseIp = "223.0." + std::to_string(groupIdx) + ".";
        if (nl1Idx == 0) {
            addr["addr"] = baseIp + std::to_string(15 + uRankId % 4 * 18); // 0/4端口对应IP
        } else {
            addr["addr"] = baseIp + std::to_string(5 + uRankId % 4 * 18);  // 1/5等端口对应IP
        }

        addr_list.push_back(addr);
    }

    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenNetLayer0RankUBX(
    TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level)
{
    json addr_list = json::array();
    json addr;
    level["net_layer"] = 0;
    string tmpNetIns = "az0-rack";
    tmpNetIns += std::to_string(superPodIdx);
    tmpNetIns += std::to_string(serverIdx);
    level["net_instance_id"] = tmpNetIns;
    level["net_type"] = "TOPO_FILE_DESC";
    level["net_addr"] = "";
    uint32_t uRankId = topoMeta[superPodIdx][serverIdx][rankIdx];
    uint32_t uPhyDevId = uRankId % UBX_DEV_NUM_PER_SERVER;

    // 1. 组内PEER2PEER
    for (uint32_t vLocalId = 0; vLocalId < rankNum; vLocalId++) {
        uint32_t vRankId = topoMeta[superPodIdx][serverIdx][vLocalId];
        auto vPhyDevId = vRankId % UBX_DEV_NUM_PER_SERVER;
        // 不同组的rank不存在peer2peer连接
        if (rankIdx == vLocalId || uPhyDevId / UBX_GROUP_SIZE != vPhyDevId / UBX_GROUP_SIZE) {
            continue;
        }
        
        addr["addr_type"] = "EID";

        string strPorts = g_uvDevice2Port[uPhyDevId][vRankId];
        auto ipAddr = g_port2IpAddr[serverIdx][uRankId].at(strPorts);
        auto eidStr = ipv4_to_128bit_id(ipAddr);
        HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, strPorts, ipAddr));

        addr["addr"] = eidStr;
        addr["ports"] = {strPorts};
        addr["plane_id"] = "plane0";
        
        addr_list.push_back(addr);
    }

    // 2. 组内PEER2NET
    const std::string US_PORTS[4] = {"0/4", "0/5", "0/6", "0/7"};
    auto peer2NetIp = g_port2IpAddr[serverIdx][uRankId].at(US_PORTS[0]); // 几个port的ip一致，随便取一个
    auto eidStr = ipv4_to_128bit_id(peer2NetIp);
    addr["addr_type"] = "EID";
    addr["addr"] = eidStr;
    addr["ports"] = US_PORTS;
    addr["plane_id"] = "planeA"; 
    addr_list.push_back(addr);
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, "0/4", peer2NetIp));
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, "0/5", peer2NetIp));
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, "0/6", peer2NetIp));
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, "0/7", peer2NetIp));

    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenNetLayer1RankUBX(
    TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level)
{
    json addr_list = json::array();
    json addr;
    level["net_layer"] = 1;
    level["net_instance_id"] = "az0";
    level["net_type"] = "CLOS";
    level["net_addr"] = "";
    uint32_t uRankId = topoMeta[superPodIdx][serverIdx][rankIdx];
    uint32_t uPhyDevId = uRankId % UBX_DEV_NUM_PER_SERVER;

    // 组外地址列表（仅d2h端口）
    json rankAddrList1 = json::array();
    const std::string D2H_PORT = "d2h";
    auto ipAddr = g_port2IpAddr[serverIdx][uPhyDevId].at(D2H_PORT);
    auto eidStr = ipv4_to_128bit_id(ipAddr);
    HCCL_VM_DEBUG("[{}] Gen UBX netlayer1 EID {}-from ip-{}", __func__, eidStr.c_str(), ipAddr.c_str());
    HCCLVM_CHK_RET(UpdatePortEidInfo(serverIdx, uPhyDevId, D2H_PORT, ipAddr));

    // d2h端口（组外ROCE通信）
    json d2hAddrNode;
    d2hAddrNode["addr_type"] = "EID";
    d2hAddrNode["addr"] = eidStr;  // 专属d2h IP
    d2hAddrNode["ports"] = {D2H_PORT};
    d2hAddrNode["plane_id"] = "plane4";  // 独立plane

    rankAddrList1.push_back(d2hAddrNode);
    level["rank_addr_list"] = rankAddrList1;

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 核心函数：生成ranktable.json
HcclVmResult DeviceTopoGenerator::InitGenRankTableJsonUBX(TopoMeta& topoMeta, uint32_t serverIdx, const std::vector<std::string>& serverIdx2Ip)
{
    uint32_t serverNum = GetServerNumFormTopoMeta(topoMeta);
    uint32_t rankNum = GetRankNumFormTopoMeta(topoMeta);

    // ===================== 核心配置 =====================
    const u32 TOTAL_RANKS = 16;          // 总16个rank
    const u32 GROUP_SIZE = 4;            // 每组4个rank
    const u32 GROUP_COUNT = TOTAL_RANKS / GROUP_SIZE;  // 4个组

    // ===================== 初始化JSON根节点 =====================
    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    // ===================== 生成rank_list =====================
    json rankList = json::array();

    uint32_t rankCnt = 0;
    for (u32 idx = 0; idx < topoMeta[0][0].size(); ++idx) {
        auto phyDevId = topoMeta[0][0][idx];
        auto rankId = rankCnt++;
        // 1. 基础rank信息
        json rankNode;
        rankNode["rank_id"] = rankId;
        rankNode["device_id"] = phyDevId;
        rankNode["local_id"] = phyDevId;

        auto devKey = phyDevId2DevKey[0].at(phyDevId);
        AddOneRank(devKey, rankId);

        auto serverKey = serverIdx2Id_[serverIdx];
        if (sim::UpdateDeviceLogicId(serverKey, phyDevId, rankId) != ACL_SUCCESS) {
            HCCL_VM_ERROR("[{}] update device logic id by physic id failed", __func__);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        // 2. 生成level_list（修正：0/4~0/7属于net_layer0，d2h属于net_layer1）
        json levelList = json::array();

        // 第一步：构建netlayer0层
        json level0;
        if (GenNetLayer0RankUBX(topoMeta, 0, 0, idx, topoMeta[0][0].size(), level0) !=
            HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("Failed to gen netlayer0 within group node! rankid: {}", phyDevId);
            continue;
        }
        levelList.push_back(level0);

        // 2.2 net_layer1：组外拓扑（仅d2h端口，ROCE）
        json level1;
        if (GenNetLayer1RankUBX(topoMeta, 0, 0, idx, topoMeta[0][0].size(), level1) !=
            HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("Failed to gen netlayer0 within group node! rankid: {}", phyDevId);
            continue;
        }
        levelList.push_back(level1);

        // 3. 填充level_list到rank节点
        rankNode["level_list"] = levelList;
        rankList.push_back(rankNode);
    }

    // ===================== 填充rank_list并写入文件 =====================
    rankTableJson["rank_list"] = rankList;

    std::ofstream topoFile(rankTableFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, rankTableFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << rankTableJson.dump(4);
    topoFile.close();
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitGenRankTableJson(TopoMeta &topoMeta, const std::string &algName)
{
    // todo: A2/A3的拓扑数据
    uint32_t serverNum = GetServerNumFormTopoMeta(topoMeta);
    uint32_t rankNum = GetRankNumFormTopoMeta(topoMeta);

    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    json rankList = json::array();
    uint32_t devCnt = 0;
    for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
        for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
            HCCLVM_CHK_RET(GenServerRanktable(topoMeta, devCnt, superPodIdx, serverIdx, algName, rankList));
        }
    }
    rankTableJson["rank_list"] = rankList;

    std::ofstream topoFile(rankTableFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, rankTableFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << rankTableJson.dump(4);
    topoFile.close();

    return HcclVmResult::HCCL_SIM_SUCCESS;
}
