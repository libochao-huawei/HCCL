#ifndef HCCL_VM_GEN_DEVICE_TOPOFILE_H
#define HCCL_VM_GEN_DEVICE_TOPOFILE_H

#include <vector>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <getopt.h>
#include <exception>
#include <json.hpp>
#include "hccl_common_defs.h"

using namespace HcclSim;
using namespace std;
using json = nlohmann::json;

class DeviceTopoGenerator {
public:
    DeviceTopoGenerator()  = default;
    ~DeviceTopoGenerator() = default;

    HcclVmResult Init(TopoMeta &topoMeta, const std::vector<std::string>& serverIdx2Ip, const std::string &algName);
    HcclVmResult InitGenTopoJson(uint32_t serverIdx);
    HcclVmResult InitGenTopoJsonMultiServer(uint32_t serverIdx);
    HcclVmResult InitGenRankTableJson(TopoMeta &topoMeta, const std::string &algName);
    HcclVmResult InitParserParserRankTableJson();
private:
    HcclVmResult InitGenTopoJsonHF(uint32_t serverIdx);
    HcclVmResult InitGenTopoJsonUBX(uint32_t serverIdx);
    HcclVmResult GenRankNetLayer0NodeTest(
        TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t localId, uint32_t rankNum, json &level);
    void InitUBXPortMapping();
    HcclVmResult GenNetLayer0RankUBX(
        TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult GenNetLayer1RankUBX(
        TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult InitGenRankTableJsonUBX(TopoMeta& topoMeta, uint32_t serverIdx, const std::vector<std::string>& serverIdx2Ip);
    HcclVmResult GenRankNetLayer1HFNode(uint32_t uRankId, json &level);
    HcclVmResult GenRankNetLayer0HFNode(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult GenRankNetHFNode(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult InitGenRankTableJsonHF(TopoMeta& topoMeta, const std::vector<std::string>& serverIdx2Ip);
    HcclVmResult GetDeviceIdAndCcuId(uint32_t logicDevId, uint8_t dieId, uint64_t &deviceKey, uint64_t &ccuKey);
    HcclVmResult UpdatePortEidInfo(uint32_t serverIdx, uint32_t phyDevId, const std::string &portName, const std::string &ipAddr);
    HcclVmResult InitDeviceInfo(uint32_t serverIdx, uint32_t rowIdx);
    HcclVmResult InitDeviceInfoMultiServer(uint32_t serverIdx);
    HcclVmResult InitHFDeviceInfo(uint32_t serverIdx, uint32_t rowIdx);
    HcclVmResult InitUBXDeviceInfo(uint32_t serverIdx);
    HcclVmResult GenXNetlayer0TopoLink(uint32_t serverIdx, uint32_t rowIdx, json &edge_list);
    HcclVmResult GenYNetlayer0TopoLink(uint32_t serverIdx, uint32_t colIdx, json &edge_list);
    HcclVmResult GenNetlayer1TopoLink(uint32_t serverIdx, uint32_t row, json &edge_list);
    HcclVmResult GenNetlayer2TopoLink(uint32_t serverIdx, uint32_t row, json &edge_list);
    HcclVmResult GenRankNetLayer0Node(
        TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult GenRankNetLayer1Node(
        TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t localId, uint32_t rankNum, json &level);
    HcclVmResult GenServerRanktable(TopoMeta &topoMeta, uint32_t &rankCnt, uint32_t superPodIdx, uint32_t serverIdx, const std::string &algName, json &rankList);
    HcclVmResult InitOnePortInfo(uint32_t serverIdx, uint64_t deviceKey, uint64_t ccuKey, uint32_t phyDevId,
        const std::string &portName, ProtocolType protocolType, uint32_t funcId);
    HcclVmResult InitSameIpPortsInfo(uint32_t serverIdx, uint64_t deviceKey, uint64_t ccuKey, uint32_t phyDevId,
        const std::vector<std::string> &portNames, ProtocolType protocolType, uint32_t funcId);
    HcclVmResult InitEndPointPairInfo(uint32_t serverIdx, uint32_t srcLogicDevId, uint32_t dstLogicDevId,
        const std::string &srcPortName, const std::string &dstPortName, bool isInServer = true);

public:
    std::vector<uint32_t> device910DXAxisRankNum_;
    std::vector<uint32_t> device910DYAxisRankNum_;
    std::vector<uint32_t> device910DRankSize_; // 910D一个server包含64个npu

private:
    std::string topoFilePath_{"topo.json"};
    std::string rankTableFilePath_{"ranktable.json"};
    std::vector<uint64_t> serverIdx2Id_{0};
};

#endif // HCCL_VM_GEN_DEVICE_TOPOFILE_H
