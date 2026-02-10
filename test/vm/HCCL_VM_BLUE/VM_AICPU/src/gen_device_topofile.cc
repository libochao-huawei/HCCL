#include <fstream>
#include <unistd.h>
#include <iostream>
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

uint64_t AddOneDevice(uint32_t physicalId)
{
    // auto runner = sim::GetCurrRunnerTls();
    // auto hostId = runner.host_id;
    // if (hostId == 0) {
    //     // not find
    //     HCCL_VM_ERROR("[AddOneDevice] wrong host id: {:d}", hostId);
    //     return 0;
    // }

    // auto host = RunnerDB::GetOneByPred<sim::Host>([hostId](const sim::Host& h) {
    //     return h.id == hostId;
    // });
    // if (!host.second) {
    //     // not find
    //     HCCL_VM_ERROR("[AddOneDevice] can not find host by key {:d}", hostId);
    //     return 0;
    // }

    sim::Device device{};
    device.server_id = 1; // todo: 后续从runner->host中获取server id
    device.physical_id = physicalId;
    device.overflow_mode = 0;
    strcpy(device.soc_version, "Ascend950");
    device.status = 0;
    return RunnerDB::Add<sim::Device>(device);
}

uint64_t AddOneServer(uint64_t podId)
{
    sim::Server server;
    server.pod_id = podId;
    return RunnerDB::Add<sim::Server>(server);
}

uint64_t AddOneHost(uint64_t serverId)
{
    sim::Host host;
    host.server_id = serverId;
    return RunnerDB::Add<sim::Host>(host);
}

uint64_t AddOneRank(uint64_t deviceKey)
{
    sim::Rank rank;
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
    HCCL_VM_DEBUG("[{}] [zhf] add ccu {} resId {}", __func__, ccuId, resId);
    return ccuId;
}

std::pair<int, int> g_ipPair = {10, 10};
std::vector<std::map<std::string, std::string>> g_port2IpAddr;
}

const static uint32_t MAX_DEV_PER_ROW = 8;
std::map<uint32_t, uint64_t> phyDevId2DevKey;
uint32_t DeviceTopoGenerator::device910DXAxisRankNum_ = 8;
uint32_t DeviceTopoGenerator::device910DYAxisRankNum_ = 8;
uint32_t DeviceTopoGenerator::device910DRankSize_ = 64; // 910D一个server包含64个npu

// todo: 原checker中algName，从哪里获取？
HcclVmResult DeviceTopoGenerator::Init(TopoMeta &topoMeta, const std::string &algName)
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
    uint32_t maxRow = 0;
    uint32_t maxCol = 0;
    for (const auto &locId : topoMeta[0][0]) {
        auto row = locId / MAX_DEV_PER_ROW;
        auto col = locId % MAX_DEV_PER_ROW;
        if (maxRow < row) {
            maxRow = row;
        }
        if (maxCol < col) {
            maxCol = col;
        }
    }
    device910DXAxisRankNum_ = maxCol + 1;
    device910DYAxisRankNum_ = maxRow + 1;
    device910DRankSize_ = device910DXAxisRankNum_ * device910DYAxisRankNum_;
    HCCL_VM_INFO("[{}] Get topo specifications, max row = {}, max col = {}", __func__, device910DYAxisRankNum_, device910DXAxisRankNum_);

    const char* hcclvmTopoType = std::getenv("HCCLVM_TOPO_TYPE");
    if (hcclvmTopoType == nullptr || std::string(hcclvmTopoType).empty()) { // 普通机型
        HCCL_VM_INFO("[{}] Enter into normal 910D topo.", __func__);
        // 初始化device、ccu和所有port信息 ———— todo: 后续根据芯片类型不同有不同的初始化
        g_port2IpAddr.resize(device910DXAxisRankNum_ * device910DYAxisRankNum_);
        for (uint32_t row = 0; row < device910DYAxisRankNum_; row++) {
            HCCLVM_CHK_RET(InitDeviceInfo(row));
        }

        // 初始化topo.json文件
        HCCLVM_CHK_RET(InitGenTopoJson());

        // 初始化ranktable.json文件 (todo: algName参数通过环境变量获取)
        HCCLVM_CHK_RET(InitGenRankTableJson(topoMeta, algName));
    } else if (std::string(hcclvmTopoType) == "HF") {
        HCCL_VM_INFO("[{}] Enter into HF 910D topo.", __func__);
        g_port2IpAddr.resize(8 * 2);
        for (uint32_t row = 0; row < 2; row++) {
            HCCLVM_CHK_RET(InitHFDeviceInfo(row));
        }

        // 初始化topo.json文件
        HCCLVM_CHK_RET(InitGenTopoJsonHF());

        // 初始化ranktable.json文件 (todo: algName参数通过环境变量获取)
        HCCLVM_CHK_RET(InitGenRankTableJsonHF(topoMeta));
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
HcclVmResult DeviceTopoGenerator::InitDeviceInfo(uint32_t rowIdx)
{
    for (uint32_t colSrc = 0; colSrc < device910DXAxisRankNum_; colSrc++) {
        uint32_t deviceId = rowIdx * device910DXAxisRankNum_ + colSrc;
        auto deviceKey = AddOneDevice(deviceId);
        auto die0Key   = AddOneCcu(deviceKey, 0);
        auto die1Key   = AddOneCcu(deviceKey, 1);
        phyDevId2DevKey[deviceId] = deviceKey;

        // 910D框内：一个ccu有7个port
        for (uint32_t idx = 0; idx < 7; idx++) {
            auto die0Port = "0/" + std::to_string(idx);
            auto die1Port = "1/" + std::to_string(idx);
            HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die0Key, deviceId, die0Port, ProtocolType::SIM_PROTOCOL_HCCS));
            HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die1Key, deviceId, die1Port, ProtocolType::SIM_PROTOCOL_HCCS));
        }
        // 910D出框：一个ccu有2个port
        HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die0Key, deviceId, "0/7", ProtocolType::SIM_PROTOCOL_ROCE));
        HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die1Key, deviceId, "0/8", ProtocolType::SIM_PROTOCOL_ROCE));
        HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die0Key, deviceId, "1/7", ProtocolType::SIM_PROTOCOL_ROCE));
        HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die1Key, deviceId, "1/8", ProtocolType::SIM_PROTOCOL_ROCE));
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitHFDeviceInfo(uint32_t rowIdx)
{
    for (uint32_t colSrc = 0; colSrc < 8; colSrc++) {
        uint32_t deviceId = rowIdx * 8 + colSrc;
        auto deviceKey = AddOneDevice(deviceId);
        auto die0Key   = AddOneCcu(deviceKey, 0);
        auto die1Key   = AddOneCcu(deviceKey, 1);
        phyDevId2DevKey[deviceId] = deviceKey;

        // 910D框内：一个ccu有7个port
        for (uint32_t idx = 0; idx < 9; idx++) {
            auto die0Port = "0/" + std::to_string(idx);
            auto die1Port = "1/" + std::to_string(idx);
            HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die0Key, deviceId, die0Port, ProtocolType::SIM_PROTOCOL_HCCS));
            HCCLVM_CHK_RET(InitOnePortInfo(deviceKey, die1Key, deviceId, die1Port, ProtocolType::SIM_PROTOCOL_HCCS));
        }
        // HF出框？
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

HcclVmResult DeviceTopoGenerator::UpdatePortEidInfo(uint32_t logicDevId, const std::string &portName, const std::string &ipAddr)
{
    // 1. 根据portName获取port
    sim::Port port{};
    HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetPortByName(logicDevId, portName, port)));

    HCCL_VM_DEBUG("[{}] zhf-add one ip addr: {}, {}", __func__, ipAddr, portName);
    auto handle =  g_rdmaHandle++;

    // // hccp_eid eid = IpToArray(ipAddr);
    // auto ipAddress = Hccl::IpAddress(ipAddr);
    // auto eid = ipAddress.GetEid();
    // std::cout<<"zhf-parse ip:: "<<ipAddr<<std::endl;
    // for (uint32_t i = 0; i < 16; i++) {
    //     std::cout<<"zhf-uuu: "<<std::hex<<static_cast<int>(eid.raw[i])<<std::endl;
    // }

    // 2. 更新Port的EID信息
    auto portKey = port.id;
    auto ret = RunnerDB::Update<sim::Port>(portKey, [portKey, handle](sim::Port &p) {
        p.rdma_handle = handle;
        p.status = 1;
        // memcpy_s(p.eid, 16 * sizeof(uint8_t), eid.raw, 16 * sizeof(uint8_t));
    });
    if (!ret) {
        HCCL_VM_ERROR("[{}] update port info failed. ipAddr={}", __func__, ipAddr);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    sim::Port p1{};
    if (GetPortByIpAddr("192.168.10.10", p1) != 0) {
        HCCL_VM_ERROR("[{}] Get dst port failed", __func__);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    HCCL_VM_DEBUG("[{}] zhf-vvvvv: {}, {:d}", __func__, p1.ip_addr, p1.rdma_handle);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitOnePortInfo(
    uint64_t deviceKey, uint64_t ccuKey, uint32_t phyDevId, const std::string &portName, ProtocolType protocolType)
{
    std:string ipAddr = "192.168.";
    if (g_ipPair.second + 1 > 100) {
        g_ipPair.first += 1;
        if (g_ipPair.first > 100) { // a.b的取值范围{10.10  ~ 100.100}
            HCCL_VM_ERROR("[{}] Create port ip addr failed. {:s}{:d}.{:d}", __func__, ipAddr, g_ipPair.first, g_ipPair.second);
            return HcclVmResult::HCCL_SIM_E_NOT_SUPPORT;
        }
        g_ipPair.second = 10;
    }
    ipAddr += std::to_string(g_ipPair.first) + "." + std::to_string(g_ipPair.second++);
    // std::cout<<"zhf-create port ip addr: "<<ipAddr<<std::endl;

    g_port2IpAddr[phyDevId][portName] = ipAddr;

    sim::Port uPortInfo{};
    uPortInfo.device_id = deviceKey;
    uPortInfo.ccu_id = ccuKey;
    uPortInfo.func_id = 2;
    strcpy(uPortInfo.name, portName.c_str());
    strcpy(uPortInfo.ip_addr, ipAddr.c_str());
    uPortInfo.protocol = protocolType;
    auto uPortId = RunnerDB::Add<sim::Port>(uPortInfo);
    
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitEndPointPairInfo(uint32_t srcPhyDevId, uint32_t dstPhyDevId, const std::string &srcPortName, const std::string &dstPortName, bool isInServer)
{
    // 1. 根据portName获取port
    sim::Port srcPort{};
    HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetPortByName(srcPhyDevId, srcPortName, srcPort)));

    sim::Port dstPort{};
    if (isInServer) {
        // 出框场景，只有srcPort，没有dstPort
        HCCLVM_CHK_RET(static_cast<HcclVmResult>(GetPortByName(dstPhyDevId, dstPortName, dstPort)));
    }

    // 2. 新增EndPointPair
    sim::EndPointPair epPair{};
    epPair.src_port = srcPort.id;
    epPair.dst_port = dstPort.id;
    HCCL_VM_DEBUG("[{}] zhf-[InitEndPointPairInfo] srcPort = {}, {} dstPort= {}, {}", __func__, srcPortName, epPair.src_port, dstPortName, epPair.dst_port);
    auto epPairId = RunnerDB::Add<sim::EndPointPair>(epPair);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenXNetlayer0TopoLink(uint32_t rowIdx, json &edge_list)
{
    vector<uint32_t> recorder(device910DXAxisRankNum_, 0);
    uint32_t linkCnt = 0;
    for (uint32_t colSrc = 0; colSrc < device910DXAxisRankNum_; colSrc++) {
        for (uint32_t colDst = colSrc + 1; colDst < device910DXAxisRankNum_; colDst++) {
            linkCnt += 1;
            std::string u_port = "0/";
            std::string v_port = "0/";

            u_port += std::to_string(recorder[colSrc]++);
            v_port += std::to_string(recorder[colDst]++);

            uint32_t uDeviceId = rowIdx * device910DXAxisRankNum_ + colSrc;
            uint32_t vDeviceId = rowIdx * device910DXAxisRankNum_ + colDst;

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
            HCCLVM_CHK_RET(InitEndPointPairInfo(uDeviceId, vDeviceId, u_port, v_port, true));
            
            AddEidInfo(uDeviceId, u_port, 0);
            AddEidInfo(vDeviceId, v_port, 0);
            g_uvDevice2Port[uDeviceId][vDeviceId] = u_port;
            g_uvDevice2Port[vDeviceId][uDeviceId] = v_port;
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenYNetlayer0TopoLink(uint32_t colIdx, json &edge_list)
{
    uint32_t linkCnt = 0;
    vector<uint32_t> recorder(device910DYAxisRankNum_, 0);
    for (uint32_t rowSrc = 0; rowSrc < device910DYAxisRankNum_; rowSrc++) {
        for (uint32_t rowDst = rowSrc + 1; rowDst < device910DYAxisRankNum_; rowDst++) {
            linkCnt += 1;
            std::string u_port = "1/";
            std::string v_port = "1/";

            u_port += std::to_string(recorder[rowSrc]++);
            v_port += std::to_string(recorder[rowDst]++);
            uint32_t uDeviceId = rowSrc * device910DYAxisRankNum_ + colIdx;
            uint32_t vDeviceId = rowDst * device910DYAxisRankNum_ + colIdx;

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
            HCCLVM_CHK_RET(InitEndPointPairInfo(uDeviceId, vDeviceId, u_port, v_port, true));

            AddEidInfo(uDeviceId, u_port, 1);
            AddEidInfo(vDeviceId, v_port, 1);
            g_uvDevice2Port[uDeviceId][vDeviceId] = u_port;
            g_uvDevice2Port[vDeviceId][uDeviceId] = v_port;
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenNetlayer1TopoLink(json &edge_list)
{
    for (uint32_t row = 0; row < device910DXAxisRankNum_; row++) {
        for (uint32_t col = 0; col < device910DYAxisRankNum_; col++) {
            uint32_t uDeviceId = row * device910DYAxisRankNum_ + col;
            std::string u_port = "0/7";
            edge_list.push_back(json{{"net_layer", 1},
                {"link_type", "PEER2NET"},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_port}},
                {"position", "DEVICE"},
                {"topo_type", "1DMESH"}});
            
            // 保存EndPointPair信息
            // HCCLVM_CHK_RET(InitEndPointPairInfo(uDeviceId, -1, u_port, {}, false));

            AddEidInfo(uDeviceId, u_port, 0);
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenNetlayer2TopoLink(json &edge_list)
{
    for (uint32_t row = 0; row < device910DXAxisRankNum_; row++) {
        for (uint32_t col = 0; col < device910DYAxisRankNum_; col++) {
            uint32_t uDeviceId = row * device910DYAxisRankNum_ + col;
            std::string u_port = "0/8";
            edge_list.push_back(json{{"net_layer", 2},
                {"link_type", "PEER2NET"},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_port}},
                {"position", "HOST"},
                {"topo_type", "1DMESH"}});
            
            // 保存EndPointPair信息
            // HCCLVM_CHK_RET(InitEndPointPairInfo(uDeviceId, -1, u_port, {}, false));

            AddEidInfo(uDeviceId, u_port, 1);
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 每行8个芯片
const static uint32_t TOPO_HF_DEV_NUM_PER_ROW = 8;
const static uint32_t DEV_NUM_PER_SERVER = 16;
HcclVmResult DeviceTopoGenerator::InitGenTopoJsonHF()
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
                    {"protocols", {"UB_CTP", "UB_MEM"}},
                    {"local_a", uDeviceId},
                    {"local_a_ports", {u_addr}},
                    {"local_b", vDeviceId},
                    {"local_b_ports", {v_addr}},
                    {"position", "DEVICE"}});
                
                // 保存EndPointPair信息
                HCCLVM_CHK_RET(InitEndPointPairInfo(uDeviceId, vDeviceId, u_addr, v_addr, true));

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
                {"protocols", {"UB_CTP", "UB_MEM"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_addr}},
                {"local_b", vDeviceId},
                {"local_b_ports", {v_addr}},
                {"position", "DEVICE"}});
            
            // 保存EndPointPair信息
            HCCLVM_CHK_RET(InitEndPointPairInfo(uDeviceId, vDeviceId, u_addr, v_addr, true));

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
HcclVmResult DeviceTopoGenerator::InitGenTopoJsonUBX()
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
        
        // 遍历组内所有节点对（src < dst，避免重复链路）
        for (uint32_t srcInGroup = 0; srcInGroup < groupSize; ++srcInGroup) {
            for (uint32_t dstInGroup = srcInGroup + 1; dstInGroup < groupSize; ++dstInGroup) {
                uint32_t uDeviceId = groupStart + srcInGroup;
                uint32_t vDeviceId = groupStart + dstInGroup;
                
                // 分配端口（1/0,1/1,1/2循环）
                std::string u_addr = "1/" + std::to_string(portIdx % 3);
                std::string v_addr = "1/" + std::to_string((portIdx + 1) % 3);
                portIdx++;

                // 添加组内链路信息
                edge_list.push_back(json{
                    {"net_layer", 0},
                    {"link_type", "PEER2PEER"},
                    {"topo_type", "1DMESH"},
                    {"topo_instance_id", 0},
                    {"protocols", {"UB_CTP", "UB_MEM"}},
                    {"local_a", uDeviceId},
                    {"local_a_ports", {u_addr}},
                    {"local_b", vDeviceId},
                    {"local_b_ports", {v_addr}},
                    {"position", "DEVICE"}
                });

                // 保存EndPointPair信息
                HCCLVM_CHK_RET(InitEndPointPairInfo(uDeviceId, vDeviceId, u_addr, v_addr, true));

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
        for (uint32_t nl1Idx = 0; nl1Idx < netLayer1Count; ++nl1Idx) {
            uint32_t nl1DeviceId = netLayer1StartId + nl1Idx;
            // 分配端口：1/4,1/5,1/6,1/7
            std::string rankPort = "1/" + std::to_string(4 + nl1Idx);
            // netlayer1节点的端口可根据实际需求调整，这里暂用相同格式
            std::string nl1Port = "1/" + std::to_string(4 + nl1Idx);

            // 添加到netlayer1节点的链路
            edge_list.push_back(json{
                {"net_layer", 1},  // 注意：net_layer设为1
                {"link_type", "PEER2PEER"},
                {"topo_type", "1DMESH"},
                {"topo_instance_id", 1},
                {"protocols", {"UB_CTP", "UB_MEM"}},
                {"local_a", rankId},
                {"local_a_ports", {rankPort}},
                {"local_b", nl1DeviceId},
                {"local_b_ports", {nl1Port}},
                {"position", "DEVICE"}
            });

            // 保存EndPointPair信息
            HCCLVM_CHK_RET(InitEndPointPairInfo(rankId, nl1DeviceId, rankPort, nl1Port, true));

            // 记录Eid信息和端口映射
            AddEidInfo(rankId, rankPort, 1);
            AddEidInfo(nl1DeviceId, nl1Port, 1);
            g_uvDevice2Port[rankId][nl1DeviceId] = rankPort;
            g_uvDevice2Port[nl1DeviceId][rankId] = nl1Port;
        }
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

HcclVmResult DeviceTopoGenerator::InitGenTopoJson()
{
    // todo: A2/A3的拓扑数据
    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "910D-2D-Fullmsh_64_plus_1";
    topoJson["peer_count"] = device910DRankSize_;

    // 创建 peer_list 数组
    json peer_list = json::array();
    for (uint32_t id = 0; id < device910DRankSize_; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 生成 edge_list
    json edge_list = json::array();

    /* ======== 生成netlayer0层级链路： X + Y轴方向 ========= */
    // X轴（Die0）
    uint32_t linkCnt = 0;
    for (uint32_t row = 0; row < device910DYAxisRankNum_; row++) {
        HCCLVM_CHK_RET(GenXNetlayer0TopoLink(row, edge_list));
    }
    // Y轴（Die1）
    for (uint32_t col = 0; col < device910DXAxisRankNum_; col++) {
        HCCLVM_CHK_RET(GenYNetlayer0TopoLink(col, edge_list));
    }
    /* ======== 生成netlayer0层级链路： X + Y轴方向 ========= */

    /* ======== 生成netlayer1 && netlayer2层级链路 ========= */
    // HCCLVM_CHK_RET(GenNetlayer1TopoLink(edge_list));
    // HCCLVM_CHK_RET(GenNetlayer2TopoLink(edge_list));
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
        if ((uRankId / 8 != vRankId / 8) && (uRankId % 8 != vRankId % 8)) {
            continue;
        }
        addr["addr_type"] = "EID";

        // string tmpAddr = "192.168.";
        // tmpAddr += std::to_string(uRankId + 1);
        // tmpAddr += ".";
        // tmpAddr += std::to_string(vRankId + 1);

        string strPorts = g_uvDevice2Port[uRankId % 64][vRankId % 64];
        auto ipAddr = g_port2IpAddr[uRankId].at(strPorts);
        // auto ipAddress = Hccl::IpAddress(ipAddr);
        // auto eid = ipAddress.GetEid();
        // auto eidStr = to_hex(eid.in6.interfaceId) + to_hex(eid.in6.subnetPrefix);
        auto eidStr = ipv4_to_128bit_id(ipAddr);
        std::cout<<"zhf-gen eid: "<<ipAddr<<": eid= "<<eidStr<<std::endl;
        addr["addr"] = eidStr;
        // sleep(20);
        // hccp_eid eid = IpStrToEID(tmpAddr);
        // SimEid simEid;
        // std::copy(std::begin(eid.raw), std::end(eid.raw), simEid.begin());
        HCCLVM_CHK_RET(UpdatePortEidInfo(uRankId % 64, strPorts, ipAddr));

        addr["ports"] = {strPorts};
        addr["plane_id"] = "planeA";
        addr_list.push_back(addr);
        
        g_devId2Ip2DieIdAndFuncId[uRankId][ipAddr] = std::pair<uint32_t, uint32_t>(
            g_devId2PortId2DieId[uRankId][strPorts], g_devId2PortId2funcId[uRankId][strPorts]);

        for (uint32_t k = 0; k < g_devId2EidInfo[uRankId].size(); k++) {
            if (g_devId2EidInfo[uRankId][k].portId == strPorts) {
                // g_devId2EidInfo[uRankId][k].eid = eid;
                g_devId2EidInfo[uRankId][k].ipAddr = ipAddr;
            }
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
    json addr;
    addr["addr_type"] = "EID";
    // string tmpAddr = "192.168.66.";
    // tmpAddr += std::to_string(rankId);
    auto ipAddr = g_port2IpAddr[rankId].at("0/7");
    auto eidStr = ipv4_to_128bit_id(ipAddr);
    std::cout<<"zhf-gen eid 1: "<<ipAddr<<": eid= "<<eidStr<<std::endl;
    // sleep(20);
    addr["addr"] = eidStr;
    addr["ports"] = {"0/7"};
    addr["plane_id"] = "plane" + std::to_string(localId);
    addr_list.push_back(addr);
    HCCL_VM_DEBUG("[{}] zhf-mmm: {}, 0/7{}", __func__, rankId, ipAddr);

    // hccp_eid eid = IpStrToEID(tmpAddr);
    // SimEid simEid;
    // std::copy(std::begin(eid.raw), std::end(eid.raw), simEid.begin());
    // HCCLVM_CHK_RET(UpdatePortEidInfo(rankId, "0/7", ipAddr));

    g_devId2Ip2DieIdAndFuncId[rankId][ipAddr] = std::pair<uint32_t, uint32_t>(
        g_devId2PortId2DieId[rankId]["0/7"], g_devId2PortId2funcId[rankId]["0/7"]);

    for (uint32_t k = 0; k < g_devId2EidInfo[rankId].size(); k++) {
        if (g_devId2EidInfo[rankId][k].portId == "0/7") {
            // g_devId2EidInfo[rankId][k].eid = eid;
            g_devId2EidInfo[rankId][k].ipAddr = ipAddr;
        }
    }

    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::GenServerRanktable(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, const std::string &algName, json &rankList)
{
    bool flag = algName == "InsAllGatherParallelMesh1DNHR" || algName == "InsAllReduceParallelMesh1DNHR" ||
                algName == "InsBroadcastParallelMesh1DNHR" || algName == "InsReduceScatterParallelMesh1DNHR" ||
                algName == "InsScatterParallelMesh1DNHR"   || algName == "InsReduceParallelMesh1DNHR" ||
                algName == "CcuAllReduceParallelMesh1DNHR" || algName == "CcuReduceScatterParallelMesh1DNHR" ||
                algName == "CcuBroadcastParallelMesh1DNHR" || algName == "CcuScatterParallelMesh1DNHR";

    // todo: 环回走哪个port？每个device都记录？
    // HCCLVM_CHK_RET(UpdatePortEidInfo(0, "0/9", "0.0.0.0"));
    uint32_t devCnt = 0;
    for (uint32_t rankIdx = 0; rankIdx < topoMeta[superPodIdx][serverIdx].size(); ++rankIdx) {
        json rank;
        auto logicDevId = devCnt++;
        rank["rank_id"] = logicDevId;
        auto phyDevId = topoMeta[superPodIdx][serverIdx][rankIdx];
        rank["device_id"] = phyDevId;
        // todo: 后续如果要支持多个超节点，这边要如何处理呢

        auto devKey = phyDevId2DevKey.at(phyDevId);
        AddOneRank(devKey);

        rank["local_id"] = phyDevId;
        if (sim::UpdateDeviceLogicId(phyDevId, logicDevId) != ACL_SUCCESS) {
            HCCL_VM_ERROR("[{}] get device by logic id 0 failed", __func__);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        json levelList = json::array();
        json level0;
        if (GenRankNetLayer0Node(topoMeta, superPodIdx, serverIdx, rankIdx, topoMeta[superPodIdx][serverIdx].size(), level0) !=
            HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("[{}] Failed to gen netlayer0 node! rankid: {}", __func__, phyDevId);
            continue;
        }
        levelList.push_back(level0);

        if (!flag) {
            rank["level_list"] = levelList;
            rankList.push_back(rank);
            continue;
        }

        json level1;
        if (GenRankNetLayer1Node(topoMeta, superPodIdx, serverIdx, rankIdx, topoMeta[superPodIdx][serverIdx].size(), level1) !=
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
        // string tmpAddr = "192.168.";
        // tmpAddr += std::to_string(uRankId + 1);
        // tmpAddr += ".";
        // tmpAddr += std::to_string(vRankId + 1);

        auto uPhyDevId = uRankId % DEV_NUM_PER_SERVER;
        auto vPhyDevId = vRankId % DEV_NUM_PER_SERVER;

        string strPorts = g_uvDevice2Port[uPhyDevId][vRankId];
        auto ipAddr = g_port2IpAddr[uRankId].at(strPorts);
        auto eidStr = ipv4_to_128bit_id(ipAddr);
        HCCL_VM_DEBUG("[{}] Gen HF EID {}-from ip-{}", __func__, eidStr.c_str(), ipAddr.c_str());
        HCCLVM_CHK_RET(UpdatePortEidInfo(uPhyDevId, strPorts, ipAddr));

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
HcclVmResult DeviceTopoGenerator::InitGenRankTableJsonHF(TopoMeta& topoMeta)
{
    uint32_t serverNum = GetServerNumFormTopoMeta(topoMeta);
    uint32_t rankNum = GetRankNumFormTopoMeta(topoMeta);

    // 初始化server和host信息 todo: 后续多server涉及runner初始化、host初始化、server初始化
    uint32_t rankCnt = 0;
    for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
        for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
            auto serverKey = AddOneServer(superPodIdx);
            auto hostKey = AddOneHost(serverKey);
            std::cout<<"[InitGenRankTableJsonHF] Add host & server: "<<serverKey<<", "<<hostKey<<std::endl;
        }
    }

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

                auto devKey = phyDevId2DevKey.at(phyDevId);
                AddOneRank(devKey);
                if (sim::UpdateDeviceLogicId(phyDevId, logicDevId) != ACL_SUCCESS) {
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

// 初始化UBX拓扑的端口映射（提前填充g_uvDevice2Port）
void DeviceTopoGenerator::InitUBXPortMapping() {
    const uint32_t groupSize = UBX_GROUP_SIZE;
    const uint32_t totalRanks = UBX_DEV_NUM_PER_SERVER;
    uint32_t portIdx = 0;

    // 1. 填充组内全连接端口（0-3,4-7,8-11,12-15每组内）
    for (uint32_t group = 0; group < totalRanks / groupSize; group++) {
        uint32_t groupStart = group * groupSize;
        for (uint32_t src = 0; src < groupSize; src++) {
            for (uint32_t dst = src + 1; dst < groupSize; dst++) {
                uint32_t uRankId = groupStart + src;
                uint32_t vRankId = groupStart + dst;
                // 组内端口：1/0,1/1,1/2循环
                std::string port = "1/" + std::to_string(portIdx % 3);
                g_uvDevice2Port[uRankId][vRankId] = port;
                g_uvDevice2Port[vRankId][uRankId] = port; // 反向映射
                portIdx++;
                
                // 初始化DieId和FuncId（示例值，可根据实际调整）
                g_devId2PortId2DieId[uRankId][port] = 0;
                g_devId2PortId2funcId[uRankId][port] = 0;
                g_devId2PortId2DieId[vRankId][port] = 1;
                g_devId2PortId2funcId[vRankId][port] = 1;
            }
        }
    }

    // 2. 填充netlayer1的端口映射（1/4,1/5,1/6,1/7）
    const uint32_t netLayer1StartId = 16; // netlayer1节点起始ID
    for (uint32_t rankId = 0; rankId < totalRanks; rankId++) {
        for (uint32_t nl1Idx = 0; nl1Idx < 4; nl1Idx++) {
            uint32_t nl1Id = netLayer1StartId + nl1Idx;
            std::string port = "1/" + std::to_string(4 + nl1Idx);
            g_uvDevice2Port[rankId][nl1Id] = port;
            g_uvDevice2Port[nl1Id][rankId] = port;
            
            // 初始化DieId和FuncId
            g_devId2PortId2DieId[rankId][port] = 2;
            g_devId2PortId2funcId[rankId][port] = 2;
        }
    }

    // 初始化EidInfo
    // for (uint32_t rankId = 0; rankId < totalRanks; rankId++) {
    //     std::vector<EidInfo> eidList;
    //     for (const auto& portPair : g_devId2PortId2DieId[rankId]) {
    //         EidInfo eid;
    //         eid.portId = portPair.first;
    //         eidList.push_back(eid);
    //     }
    //     g_devId2EidInfo[rankId] = eidList;
    // }
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
        // std::string baseIp = "223.0." + std::to_string(groupIdx) + ".";
        // addr["addr"] = baseIp + std::to_string(10 + vRankId % 4 * 18); // 匹配示例IP规律
        auto ipAddr = g_port2IpAddr[uRankId].at(strPorts);
        auto eidStr = ipv4_to_128bit_id(ipAddr);
        HCCL_VM_DEBUG("[{}] Gen HF EID {}-from ip-{}", __func__, eidStr.c_str(), ipAddr.c_str());
        HCCLVM_CHK_RET(UpdatePortEidInfo(uPhyDevId, strPorts, ipAddr));
        
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
        // if (nl1Idx == 0) {
        //     addr["addr"] = baseIp + std::to_string(15 + uRankId % 4 * 18); // 0/4端口对应IP
        // } else {
        //     addr["addr"] = baseIp + std::to_string(5 + uRankId % 4 * 18);  // 1/5等端口对应IP
        // }
        if (nl1Idx == 0) {
            addr["addr"] = baseIp + std::to_string(15 + uRankId % 4 * 18); // 0/4端口对应IP
        } else {
            addr["addr"] = baseIp + std::to_string(5 + uRankId % 4 * 18);  // 1/5等端口对应IP
        }

        // auto ipAddr = g_port2IpAddr[uRankId].at(strPorts);
        // auto eidStr = ipv4_to_128bit_id(ipAddr);
        // HCCL_VM_DEBUG("[{}] Gen HF EID {}-from ip-{}", __func__, eidStr.c_str(), ipAddr.c_str());
        // HCCLVM_CHK_RET(UpdatePortEidInfo(uPhyDevId, strPorts, ipAddr));

        addr_list.push_back(addr);

        // 填充全局映射表
        // g_devId2Ip2DieIdAndFuncId[uRankId][IpAddress(addr["addr"].get<std::string>()).ip] = 
        //     std::pair<uint32_t, uint32_t>(
        //         g_devId2PortId2DieId[uRankId][strPorts], 
        //         g_devId2PortId2funcId[uRankId][strPorts]
        //     );

        // 更新EidInfo的IP地址
        for (uint32_t k = 0; k < g_devId2EidInfo[uRankId].size(); k++) {
            if (g_devId2EidInfo[uRankId][k].portId == strPorts) {
                // g_devId2EidInfo[uRankId][k].ipAddress = IpAddress(addr["addr"].get<std::string>());
            }
        }
    }

    level["rank_addr_list"] = addr_list;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 核心函数：生成ranktable.json
HcclVmResult DeviceTopoGenerator::InitGenRankTableJsonUBX(TopoMeta& topoMeta)
{
    // 初始化UBX拓扑的端口映射
    InitUBXPortMapping();

    uint32_t serverNum = GetServerNumFormTopoMeta(topoMeta);
    uint32_t rankNum = GetRankNumFormTopoMeta(topoMeta);

    // 初始化server和host信息 todo: 后续多server涉及runner初始化、host初始化、server初始化
    uint32_t rankCnt = 0;
    for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
        for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
            auto serverKey = AddOneServer(superPodIdx);
            auto hostKey = AddOneHost(serverKey);
            std::cout<<"[InitGenRankTableJsonUBX] Add host & server: "<<serverKey<<", "<<hostKey<<std::endl;
        }
    }

    // 基础JSON结构
    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    uint32_t rankId = 0;
    json rankList = json::array();

    // 遍历所有rank生成rank_list
    for (uint32_t i = 0; i < topoMeta.size(); ++i) { // superPodIdx
        for (uint32_t j = 0; j < topoMeta[i].size(); ++j) { // serverIdx
            for (uint32_t k = 0; k < topoMeta[i][j].size(); ++k) { // rankIdx
                json rank;
                rank["rank_id"] = rankId++;
                rank["device_id"] = topoMeta[i][j][k];
                rank["local_id"] = topoMeta[i][j][k] % 4; // local_id按组内索引（0-3）

                // 生成level_list（包含netlayer0和netlayer1）
                json levelList = json::array();
                
                // 生成netlayer0（组内全连接）
                json level0;
                if (GenRankNetLayer0HFNode(topoMeta, i, j, k, topoMeta[i][j].size(), level0) !=
                    HcclVmResult::HCCL_SIM_SUCCESS) {
                    // HCCL_ERROR("Failed to gen netlayer0 node! rankid: %d", topoMeta[i][j][k]);
                    continue;
                }
                levelList.push_back(level0);

                // 生成netlayer1（CLOS网络）
                json level1;
                uint32_t uRankId = topoMeta[i][j][k];
                if (GenRankNetLayer1HFNode(uRankId, level1) != HcclVmResult::HCCL_SIM_SUCCESS) {
                    // HCCL_ERROR("Failed to gen netlayer1 node! rankid: %d", uRankId);
                    continue;
                }
                levelList.push_back(level1);

                rank["level_list"] = levelList;
                rankList.push_back(rank);
            }
        }
    }

    rankTableJson["rank_list"] = rankList;

    // // 转换为格式化字符串
    // std::ostringstream oss;
    // oss << rankTableJson.dump(4);
    // rankTableString = oss.str();

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitGenRankTableJson(TopoMeta &topoMeta, const std::string &algName)
{
    // todo: A2/A3的拓扑数据
    uint32_t serverNum = GetServerNumFormTopoMeta(topoMeta);
    uint32_t rankNum = GetRankNumFormTopoMeta(topoMeta);

    // 初始化server和host信息 todo: 后续多server涉及runner初始化、host初始化、server初始化
    uint32_t rankCnt = 0;
    for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
        for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
            auto serverKey = AddOneServer(superPodIdx);
            auto hostKey = AddOneHost(serverKey);
            std::cout<<"[InitGenRankTableJson] Add host & server: "<<serverKey<<", "<<hostKey<<std::endl;
        }
    }

    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    json rankList = json::array();
    for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
        for (uint32_t serverIdx = 0; serverIdx < topoMeta[superPodIdx].size(); ++serverIdx) {
            HCCLVM_CHK_RET(GenServerRanktable(topoMeta, superPodIdx, serverIdx, algName, rankList));
        }
    }
    rankTableJson["rank_list"] = rankList;

    if (access(rankTableFilePath_.c_str(), F_OK) == 0) {
        std::cout<<"[InitGenRankTableJson] rankTable exit do nothing :"<<rankTableFilePath_<<std::endl;
        InitParserParserRankTableJson();
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    std::ofstream topoFile(rankTableFilePath_);
    if (!topoFile.is_open()) {
        HCCL_VM_ERROR("[{}] Failed to open file {}", __func__, rankTableFilePath_);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    topoFile << rankTableJson.dump(4);
    topoFile.close();

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DeviceTopoGenerator::InitParserParserRankTableJson()
{
    try {
        std::ifstream rankTableFile(rankTableFilePath_);
        if (!rankTableFile.is_open()) {
            throw std::runtime_error("Failed to open config file: " + rankTableFilePath_);
        }
        nlohmann::json data = nlohmann::json::parse(rankTableFile);
        // 解析服务器列表
        if (data.contains("server_list") && data["server_list"].is_array()) {
            for (const auto& server : data["server_list"]) {
                // 解析设备列表
                if (server.contains("device") && server["device"].is_array()) {
                    for (const auto& device : server["device"]) {
                        uint32_t device_id = static_cast<uint32_t>(std::stoul(device.value("device_id", "0")));
                        uint32_t super_device_id = static_cast<uint32_t>(std::stoul(device.value("super_device_id", "0")));
                        printf("[InitParserParserRankTableJson] update device id:%u, super device id:%u\n", device_id, super_device_id);
                        if (sim::UpdateSuperDeviceId(device_id, super_device_id) != ACL_SUCCESS) {
                            printf("[ERROR]get device by logic id 0 failed.");
                            return HcclVmResult::HCCL_SIM_E_INTERNAL;
                        }
                    }
                }
            }
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "General error: " << e.what() << std::endl;
    }
    return HcclVmResult::HCCL_SIM_E_UNAVAIL;
}