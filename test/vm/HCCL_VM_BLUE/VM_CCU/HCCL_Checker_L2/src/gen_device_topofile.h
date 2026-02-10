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

    HcclVmResult Init(TopoMeta &topoMeta, const std::string &algName);
    HcclVmResult InitGenTopoJson();
    HcclVmResult InitGenRankTableJson(TopoMeta &topoMeta, const std::string &algName);

private:
    HcclVmResult InitGenTopoJsonHF();
    HcclVmResult InitGenTopoJsonUBX();
    void InitUBXPortMapping();
    HcclVmResult InitGenRankTableJsonUBX(TopoMeta& topoMeta);
    HcclVmResult GenRankNetLayer1HFNode(uint32_t uRankId, json &level);
    HcclVmResult GenRankNetLayer0HFNode(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult GenRankNetHFNode(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult InitGenRankTableJsonHF(TopoMeta& topoMeta);
    HcclVmResult GetDeviceIdAndCcuId(uint32_t logicDevId, uint8_t dieId, uint64_t &deviceKey, uint64_t &ccuKey);
    HcclVmResult UpdatePortEidInfo(uint32_t logicDevId, const std::string &portName, const std::string &ipAddr);
    HcclVmResult InitDeviceInfo(uint32_t rowIdx);
    HcclVmResult InitHFDeviceInfo(uint32_t rowIdx);
    HcclVmResult GenXNetlayer0TopoLink(uint32_t rowIdx, json &edge_list);
    HcclVmResult GenYNetlayer0TopoLink(uint32_t colIdx, json &edge_list);
    HcclVmResult GenNetlayer1TopoLink(json &edge_list);
    HcclVmResult GenNetlayer2TopoLink(json &edge_list);
    HcclVmResult GenRankNetLayer0Node(
        TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t rankIdx, uint32_t rankNum, json &level);
    HcclVmResult GenRankNetLayer1Node(
        TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, uint32_t localId, uint32_t rankNum, json &level);
    HcclVmResult GenServerRanktable(TopoMeta &topoMeta, uint32_t superPodIdx, uint32_t serverIdx, const std::string &algName, json &rankList);
    HcclVmResult InitOnePortInfo(uint64_t deviceId, uint64_t ccuId, uint32_t funcId, const std::string &portName, ProtocolType protocolType);
    HcclVmResult InitEndPointPairInfo(uint32_t srcLogicDevId, uint32_t dstLogicDevId, const std::string &srcPortName, const std::string &dstPortName, bool isInServer = true);

public:
    static uint32_t device910DXAxisRankNum_;
    static uint32_t device910DYAxisRankNum_;
    static uint32_t device910DRankSize_; // 910D一个server包含64个npu

private:
    std::string topoFilePath_{"topo.json"};
    std::string rankTableFilePath_{"ranktable.json"};
};

#endif // HCCL_VM_GEN_DEVICE_TOPOFILE_H
