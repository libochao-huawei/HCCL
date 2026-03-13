#include "coll_gen_json.h"

#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>

#include "llt_common.h"

namespace Hccl {
using json = nlohmann::json;

std::map<u32, std::vector<HrtDevEidInfo>> g_devId2EidInfo;
std::map<u32, std::map<IpAddress, std::pair<uint32_t, uint32_t>>> g_devId2Ip2DieIdAndFuncId;
std::map<u32, u32> g_devId2funcId;
std::map<u32, map<std::string, u32>> g_devId2PortId2DieId;
std::map<u32, map<std::string, u32>> g_devId2PortId2funcId;
std::map<u32, map<std::string, HrtDevEidInfo>> g_devId2PortId2EidInfo;
std::map<u32, map<u32, string>> g_uvDevice2Port;

void AddEidInfo(u32 deviceId, std::string addr, u32 dieId)
{
    HrtDevEidInfo eidInfo;
    eidInfo.portId = addr;
    eidInfo.dieId = dieId;
    eidInfo.chipId = deviceId;
    if (g_devId2funcId.count(deviceId) == 0) {
        g_devId2funcId[deviceId] = 1;
    }
    eidInfo.funcId = g_devId2funcId[deviceId] / 7;
    g_devId2funcId[deviceId] += 1;
    g_devId2EidInfo[deviceId].push_back(eidInfo);
    g_devId2PortId2DieId[deviceId][addr] = dieId;
    g_devId2PortId2funcId[deviceId][addr] = eidInfo.funcId;
    return;
}

HcclResult InitGenRankTableJsonV1(TopoMeta& topoMeta, std::string& rankTableString)
{
    u32 serverNum = GetServerNumFormTopoMeta(topoMeta);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);

    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    json rankList = json::array();
    const char* dieNum = std::getenv("HCCL_IODIE_NUM");
    if (dieNum != nullptr) {
        std::string value(dieNum);
        if (value == "2" ) {
            for (u32 i = 0; i < rankNum; ++i) {
                json rank;
                rank["rank_id"] = i;
                rank["device_id"] = topoMeta[0][0][i];
                // 后续如果要支持多个超节点，这边要如何处理呢
                rank["local_id"] = topoMeta[0][0][i];

                json levelList = json::array();
                json addr_list = json::array();
                json level;
                json addr;
                level["net_layer"] = 0;
                level["net_instance_id"] = "az0-rack0";
                level["net_type"] = "TOPO_FILE_DESC";

                u32 tmpPort = 1;

                for (u32 j = 0; j < rankNum; j++) {
                    if (i == j) {
                        continue;
                    }
                    if ((topoMeta[0][0][i] / 8 != topoMeta[0][0][j] / 8) && (topoMeta[0][0][i] % 8 != topoMeta[0][0][j] % 8)) {
                        continue;
                    }
                    addr["addr_type"] = "IPV4";
                    string tmpAddr = "192.168.";
                    tmpAddr += std::to_string(i + 1);
                    tmpAddr += ".";
                    tmpAddr += std::to_string(j + 1);
                    addr["addr"] = tmpAddr;
                    string strPorts = g_uvDevice2Port[topoMeta[0][0][i]][topoMeta[0][0][j]];
                    addr["ports"] = {strPorts};
                    addr["plane_id"] = "planeA";
                    addr_list.push_back(addr);
                    g_devId2Ip2DieIdAndFuncId[topoMeta[0][0][i]][IpAddress(tmpAddr)] = std::pair<uint32_t, uint32_t>(
                        g_devId2PortId2DieId[topoMeta[0][0][i]][strPorts], g_devId2PortId2funcId[topoMeta[0][0][i]][strPorts]);

                    for (u32 k = 0; k < g_devId2EidInfo[topoMeta[0][0][i]].size(); k++) {
                        if ( g_devId2EidInfo[topoMeta[0][0][i]][k].portId == strPorts) {
                            g_devId2EidInfo[topoMeta[0][0][i]][k].ipAddress = IpAddress(tmpAddr);
                        }
                    }
                }
                level["rank_addr_list"] = addr_list;
                levelList.push_back(level);

                rank["level_list"] = levelList;
                rankList.push_back(rank);
            }
            rankTableJson["rank_list"] = rankList;
        }
    } else {
        for (u32 i = 0; i < rankNum; ++i) {
            json rank;
            rank["rank_id"] = i;
            rank["device_id"] = i;
            // 后续如果要支持多个超节点，这边要如何处理呢
            rank["local_id"] = topoMeta[0][0][i];

            json levelList = json::array();
            json addr_list = json::array();
            json level;
            json addr;
            level["net_layer"] = 0;
            level["net_instance_id"] = "az0-rack0";
            level["net_type"] = "TOPO_FILE_DESC";

            for (u32 j = 0; j < rankNum; j++) {
                if (i == j) {
                    continue;
                }

                addr["addr_type"] = "IPV4";
                string tmpAddr = "192.168.";
                tmpAddr += std::to_string(i + 1);
                tmpAddr += ".";
                tmpAddr += std::to_string(j + 1);
                addr["addr"] = tmpAddr;
                string strPorts = g_uvDevice2Port[topoMeta[0][0][i]][topoMeta[0][0][j]];
                addr["ports"] = {strPorts};
                addr["plane_id"] = "planeA";
                addr_list.push_back(addr);
                g_devId2Ip2DieIdAndFuncId[i][IpAddress(tmpAddr)] = std::pair<uint32_t, uint32_t>(
                    g_devId2PortId2DieId[i][strPorts], g_devId2PortId2funcId[i][strPorts]);

                for (u32 k = 0; k < g_devId2EidInfo[i].size(); k++) {
                    if ( g_devId2EidInfo[i][k].portId == strPorts) {
                        g_devId2EidInfo[i][k].ipAddress = IpAddress(tmpAddr);
                    }
                }
            }

            level["rank_addr_list"] = addr_list;
            levelList.push_back(level);

            rank["level_list"] = levelList;
            rankList.push_back(rank);
        }
        rankTableJson["rank_list"] = rankList;
    }
    std::ostringstream oss;
    oss << rankTableJson.dump(4);
    rankTableString = oss.str();


    return HcclResult::HCCL_SUCCESS;
}

HcclResult GenRankNetHFNode(TopoMeta &topoMeta, u32 superPodIdx, u32 serverIdx, u32 rankIdx, u32 rankNum, json &level)
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
    u32 uRankId = topoMeta[superPodIdx][serverIdx][rankIdx];

    for (u32 vLocalId = 0; vLocalId < rankNum; vLocalId++) {
        if (rankIdx == vLocalId) {
            continue;
        }
        u32 vRankId = topoMeta[superPodIdx][serverIdx][vLocalId];
        addr["addr_type"] = "IPV4";
        string tmpAddr = "192.168.";
        tmpAddr += std::to_string(uRankId + 1);
        tmpAddr += ".";
        tmpAddr += std::to_string(vRankId + 1);
        addr["addr"] = tmpAddr;
        string strPorts = g_uvDevice2Port[uRankId % 64][vRankId % 64];
        addr["ports"] = {strPorts};
        if (vLocalId / 8 == rankIdx / 8) {
            addr["plane_id"] = "plane0";
        } else {
            addr["plane_id"] = "plane1";
        }
        
        addr_list.push_back(addr);
        g_devId2Ip2DieIdAndFuncId[uRankId][IpAddress(tmpAddr)] = std::pair<uint32_t, uint32_t>(
            g_devId2PortId2DieId[uRankId][strPorts], g_devId2PortId2funcId[uRankId][strPorts]);

        for (u32 k = 0; k < g_devId2EidInfo[uRankId].size(); k++) {
            if (g_devId2EidInfo[uRankId][k].portId == strPorts) {
                g_devId2EidInfo[uRankId][k].ipAddress = IpAddress(tmpAddr);
            }
        }
    }
    level["rank_addr_list"] = addr_list;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InitGenRankTableJsonHF(TopoMeta& topoMeta, std::string& rankTableString)
{
    u32 serverNum = GetServerNumFormTopoMeta(topoMeta);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);

    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    u32 rankId = 0;
    json rankList = json::array();
    for (u32 i = 0; i < topoMeta.size(); ++i) {
        for (u32 j = 0; j < topoMeta[i].size(); ++j) {
            for (u32 k = 0; k < topoMeta[i][j].size(); ++k) {
                json rank;
                rank["rank_id"] = rankId++;
                rank["device_id"] = topoMeta[i][j][k];
                // 后续如果要支持多个超节点，这边要如何处理呢
                rank["local_id"] = topoMeta[i][j][k];

                json levelList = json::array();
                json level0;
                if (GenRankNetHFNode(topoMeta, i, j, k, topoMeta[i][j].size(), level0) !=
                    HcclResult::HCCL_SUCCESS) {
                    HCCL_ERROR("Failed to gen netlayer0 node! rankid: %d", topoMeta[i][j][k]);
                    continue;
                }
                levelList.push_back(level0);
                rank["level_list"] = levelList;
                rankList.push_back(rank);
            }
        }
    }
    rankTableJson["rank_list"] = rankList;

    std::ostringstream oss;
    oss << rankTableJson.dump(4);
    rankTableString = oss.str();

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InitGenTopoJsonV1(std::string &topoFileName, bool is1DTopo)
{
    u32 rankSizeNum = 64;    // 需要更大ranksize时，需要修改该值 = 64，最大支持64

    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "910D-2D-Fullmsh_64_plus_1";
    topoJson["peer_count"] = rankSizeNum;

    // 创建 peer_list 数组
    json peer_list = json::array();
    for (u32 id = 0; id < rankSizeNum; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 生成 edge_list
    json edge_list = json::array();

    // 生成行方向，即die0上的链路
    u32 linkCnt = 0;
    for (u32 row = 0; row < 8; row++) {
        vector<u32> recorder(8, 1);
        linkCnt = 0;
        for (u32 colSrc = 0; colSrc < 8; colSrc++) {
            u32 tmpDst = colSrc + 1;
            for (u32 colDst = colSrc + 1; colDst < 8; colDst++) {
                linkCnt += 1;
                std::string u_addr = "0/";
                std::string v_addr = "0/";

                u_addr += std::to_string(recorder[colSrc]++);
                v_addr += std::to_string(recorder[colDst]++);

                u32 uDeviceId = row * 8 + colSrc;
                u32 vDeviceId = row * 8 + colDst;

                edge_list.push_back(json{
                    {"net_layer", 0},
                    {"link_type", "PEER2PEER"},
                    {"protocols", {"UB_CTP"}},
                    {"local_a", uDeviceId},
                    {"local_a_ports", {u_addr}},
                    {"local_b", vDeviceId},
                    {"local_b_ports", {v_addr}},
                    {"position", "DEVICE"}
                });
                AddEidInfo(uDeviceId, u_addr, 0);
                AddEidInfo(vDeviceId, v_addr, 0);
                g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
                g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
            }
        }
    }

    if (!is1DTopo) {
        // 生成列方向，即die1上的链路
        for (u32 col = 0; col < 8; col++) {
            linkCnt = 0;
            vector<u32> recorder(8, 1);
            for (u32 rowSrc = 0; rowSrc < 8; rowSrc++) {
                u32 tmpDst = rowSrc + 1;
                for (u32 rowDst = rowSrc + 1; rowDst < 8; rowDst++) {
                    linkCnt += 1;
                    std::string u_addr = "1/";
                    std::string v_addr = "1/";

                    u_addr += std::to_string(recorder[rowSrc]++);
                    v_addr += std::to_string(recorder[rowDst]++);
                    u32 uDeviceId = rowSrc * 8 + col;
                    u32 vDeviceId = rowDst * 8 + col;

                    edge_list.push_back(json{
                        {"net_layer", 0},
                        {"link_type", "PEER2PEER"},
                        {"protocols", {"UB_CTP"}},
                        {"local_a", uDeviceId},
                        {"local_a_ports", {u_addr}},
                        {"local_b", vDeviceId},
                        {"local_b_ports", {v_addr}},
                        {"position", "DEVICE"}
                    });

                    AddEidInfo(uDeviceId, u_addr, 1);
                    AddEidInfo(vDeviceId, v_addr, 1);
                    g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
                    g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
                }
            }
        }
    }

    topoJson["edge_count"] = edge_list.size();
    topoJson["edge_list"] = edge_list;

    const char* value = std::getenv("TOPO_PATH_NAME");
    if (value != nullptr) {
        std::string prefix(value);
        topoFileName = prefix + "_topo.json";
    } else {
        topoFileName = "topo.json";
    }

    std::ofstream topoFile(topoFileName);
    if (!topoFile.is_open()) {
        HCCL_ERROR("Failed to open file %s", topoFileName.c_str());
        return HcclResult::HCCL_E_PARA;
    }

    topoFile << topoJson.dump(4);
    topoFile.close();

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InitGenTopoJsonHF(std::string &topoFileName, bool is1DTopo)
{
    u32 rankSizeNum = 16;  // 需要更大ranksize时，需要修改该值 = 64，最大支持64

    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "Atlas 550";
    topoJson["peer_count"] = rankSizeNum;

    // 创建 peer_list 数组
    json peer_list = json::array();
    for (u32 id = 0; id < rankSizeNum; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 生成 edge_list
    json edge_list = json::array();

    // 生成行方向，即die1上的链路
    u32 linkCnt = 0;
    for (u32 row = 0; row < 2; row++) {
        vector<u32> recorder(8, 1);
        linkCnt = 0;
        for (u32 colSrc = 0; colSrc < 8; colSrc++) {
            for (u32 colDst = colSrc + 1; colDst < 8; colDst++) {
                linkCnt += 1;
                std::string u_addr = "1/";
                std::string v_addr = "1/";

                u_addr += std::to_string(recorder[colSrc]);
                v_addr += std::to_string(8 - recorder[colSrc]);
                recorder[colSrc]++;
                u32 uDeviceId = row * 8 + colSrc;
                u32 vDeviceId = row * 8 + colDst;

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
                AddEidInfo(uDeviceId, u_addr, 0);
                AddEidInfo(vDeviceId, v_addr, 0);
                g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
                g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
            }
        }
    }

    // 生成列方向，即die1上的链路
    vector<u32> recorder1(8, 1);
    vector<u32> recorder2(8, 1);
    for (u32 colSrc = 0; colSrc < 8; colSrc++) {
        for (u32 colDst = 0; colDst < 8; colDst++) {
            std::string u_addr = "0/";
            std::string v_addr = "0/";

            u_addr += std::to_string(recorder1[colSrc]++);
            v_addr += std::to_string(recorder2[colDst]++);
            u32 uDeviceId = colSrc;
            u32 vDeviceId = colDst + 8;

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

            AddEidInfo(uDeviceId, u_addr, 1);
            AddEidInfo(vDeviceId, v_addr, 1);
            g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
            g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
        }
    }

    topoJson["edge_count"] = edge_list.size();
    topoJson["edge_list"] = edge_list;

    const char *value = std::getenv("TOPO_PATH_NAME");
    if (value != nullptr) {
        std::string prefix(value);
        topoFileName = prefix + "_topo.json";
    } else {
        topoFileName = "topo.json";
    }

    std::ofstream topoFile(topoFileName);
    if (!topoFile.is_open()) {
        HCCL_ERROR("Failed to open file %s", topoFileName.c_str());
        return HcclResult::HCCL_E_PARA;
    }

    topoFile << topoJson.dump(4);
    topoFile.close();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GenRankNetLayer0Node(
    TopoMeta &topoMeta, u32 superPodIdx, u32 serverIdx, u32 rankIdx, u32 rankNum, json &level)
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
    u32 uRankId = topoMeta[superPodIdx][serverIdx][rankIdx];

    for (u32 vLocalId = 0; vLocalId < rankNum; vLocalId++) {
        if (rankIdx == vLocalId) {
            continue;
        }
        u32 vRankId = topoMeta[superPodIdx][serverIdx][vLocalId];
        if ((uRankId / 8 != vRankId / 8) && (uRankId % 8 != vRankId % 8)) {
            continue;
        }
        addr["addr_type"] = "IPV4";
        string tmpAddr = "192.168.";
        tmpAddr += std::to_string(uRankId + 1);
        tmpAddr += ".";
        tmpAddr += std::to_string(vRankId + 1);
        addr["addr"] = tmpAddr;
        string strPorts = g_uvDevice2Port[uRankId % 64][vRankId % 64];
        addr["ports"] = {strPorts};
        addr["plane_id"] = "planeA";
        addr_list.push_back(addr);
        g_devId2Ip2DieIdAndFuncId[uRankId][IpAddress(tmpAddr)] = std::pair<uint32_t, uint32_t>(
            g_devId2PortId2DieId[uRankId][strPorts], g_devId2PortId2funcId[uRankId][strPorts]);

        for (u32 k = 0; k < g_devId2EidInfo[uRankId].size(); k++) {
            if (g_devId2EidInfo[uRankId][k].portId == strPorts) {
                g_devId2EidInfo[uRankId][k].ipAddress = IpAddress(tmpAddr);
            }
        }
    }
    level["rank_addr_list"] = addr_list;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GenRankNetLayer1Node(
    TopoMeta &topoMeta, u32 superPodIdx, u32 serverIdx, u32 localId, u32 rankNum, json &level)
{
    level["net_layer"] = 1;
    level["net_instance_id"] = "az0";
    level["net_type"] = "CLOS";
    level["net_addr"] = "";
    u32 rankId = topoMeta[superPodIdx][serverIdx][localId];

    json addr_list = json::array();
    json addr;
    addr["addr_type"] = "IPV4";
    string tmpAddr = "192.168.66.";
    tmpAddr += std::to_string(rankId * 2);
    addr["addr"] = tmpAddr;
    addr["ports"] = {"0/7"};
    addr_list.push_back(addr);

    g_devId2Ip2DieIdAndFuncId[rankId][IpAddress(tmpAddr)] = std::pair<uint32_t, uint32_t>(
        g_devId2PortId2DieId[rankId]["0/7"], g_devId2PortId2funcId[rankId]["0/7"]);

    for (u32 k = 0; k < g_devId2EidInfo[rankId].size(); k++) {
        if (g_devId2EidInfo[rankId][k].portId == "0/7") {
            g_devId2EidInfo[rankId][k].ipAddress = IpAddress(tmpAddr);
        }
    }
    
    json addr_die1;
    addr_die1["addr_type"] = "IPV4";
    string tmpAddr_die1 = "192.168.68.";
    tmpAddr_die1 += std::to_string(rankId * 2 + 1);
    addr_die1["addr"] = tmpAddr_die1;
    addr_die1["ports"] = {"1/7"};
    addr_die1["plane_id"] = "plane" + std::to_string(localId * 2 + 1);
    addr_list.push_back(addr_die1);
    
    g_devId2Ip2DieIdAndFuncId[rankId][IpAddress(tmpAddr_die1)] = std::pair<uint32_t, uint32_t>(
        g_devId2PortId2DieId[rankId]["1/7"], g_devId2PortId2funcId[rankId]["1/7"]);

    for (u32 k = 0; k < g_devId2EidInfo[rankId].size(); k++) {
        if (g_devId2EidInfo[rankId][k].portId == "1/7") {
            g_devId2EidInfo[rankId][k].ipAddress = IpAddress(tmpAddr_die1);
        }
    }


    level["rank_addr_list"] = addr_list;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GenRankNetLayer2Node(
    TopoMeta &topoMeta, u32 superPodIdx, u32 serverIdx, u32 localId, u32 rankNum, json &level)
{
    level["net_layer"] = 2;
    level["net_instance_id"] = "superpod_0";
    level["net_type"] = "CLOS";
    level["net_addr"] = "";
    u32 rankId = topoMeta[superPodIdx][serverIdx][localId];

    json addr_list = json::array();
    json addr;
    addr["addr_type"] = "IPV4";
    string tmpAddr = "192.168.67.";
    tmpAddr += std::to_string(rankId);
    addr["addr"] = tmpAddr;
    addr["ports"] = {"0/8"};
    addr["plane_id"] = "plane0";
    addr_list.push_back(addr);

    level["rank_addr_list"] = addr_list;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InitGenRankTableJson(TopoMeta &topoMeta, const CheckerOpParam& param, std::string &rankTableString)
{
    if (param.devtype == CheckerDevType::DEV_TYPE_HF) {
        InitGenRankTableJsonHF(topoMeta, rankTableString);
        return HCCL_SUCCESS;
    }
    if(param.devtype != CheckerDevType::DEV_TYPE_910_95) {
        return InitGenRankTableJsonV1(topoMeta, rankTableString);
    }
    u32 serverNum = GetServerNumFormTopoMeta(topoMeta);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);

    json rankTableJson;
    rankTableJson["version"] = "2.0";
    rankTableJson["rank_count"] = rankNum;

    bool hasLevel1 = !topoMeta.empty() && (topoMeta[0].size() > 1);
    u32 rankId = 0;
    json rankList = json::array();
    for (u32 i = 0; i < topoMeta.size(); ++i) {
        for (u32 j = 0; j < topoMeta[i].size(); ++j) {
            for (u32 k = 0; k < topoMeta[i][j].size(); ++k) {
                json rank;
                rank["rank_id"] = rankId++;
                rank["device_id"] = topoMeta[i][j][k];
                // 后续如果要支持多个超节点，这边要如何处理呢
                rank["local_id"] = topoMeta[i][j][k];

                json levelList = json::array();
                json level0;
                if (GenRankNetLayer0Node(topoMeta, i, j, k, topoMeta[i][j].size(), level0) !=
                    HcclResult::HCCL_SUCCESS) {
                    HCCL_ERROR("Failed to gen netlayer0 node! rankid: %d", topoMeta[i][j][k]);
                    continue;
                }
                levelList.push_back(level0);

                if (!hasLevel1) {
                    rank["level_list"] = levelList;
                    rankList.push_back(rank);
                    continue;
                }

                json level1;
                if (GenRankNetLayer1Node(topoMeta, i, j, k, topoMeta[i][j].size(), level1) !=
                    HcclResult::HCCL_SUCCESS) {
                    HCCL_ERROR("Failed to gen netlayer1 node! rankid: %d", topoMeta[i][j][k]);
                    continue;
                }
                levelList.push_back(level1);

                rank["level_list"] = levelList;
                rankList.push_back(rank);
            }
        }
    }
    rankTableJson["rank_list"] = rankList;

    std::ostringstream oss;
    oss << rankTableJson.dump(4);
    rankTableString = oss.str();

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InitGenTopoJson(std::string &topoFileName, const CheckerOpParam& param, bool is1DTopo)
{
    if (param.devtype == CheckerDevType::DEV_TYPE_HF) {
        InitGenTopoJsonHF(topoFileName, is1DTopo);
        return HCCL_SUCCESS;
    }
    if(param.devtype !=  CheckerDevType::DEV_TYPE_910_95) {
        return InitGenTopoJsonV1(topoFileName, is1DTopo);
    }
    u32 rankSizeNum = 64;  // 需要更大ranksize时，需要修改该值 = 64，最大支持64

    json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardware_type"] = "910D-2D-Fullmsh_64_plus_1";
    topoJson["peer_count"] = rankSizeNum;

    // 创建 peer_list 数组
    json peer_list = json::array();
    for (u32 id = 0; id < rankSizeNum; ++id) {
        peer_list.push_back(json{{"local_id", id}});
    }
    topoJson["peer_list"] = peer_list;

    // 生成 edge_list
    json edge_list = json::array();

    // 生成行方向，即die0上的链路
    u32 linkCnt = 0;
    for (u32 row = 0; row < 8; row++) {
        vector<u32> recorder(8, 1);
        linkCnt = 0;
        for (u32 colSrc = 0; colSrc < 8; colSrc++) {
            u32 tmpDst = colSrc + 1;
            for (u32 colDst = colSrc + 1; colDst < 8; colDst++) {
                linkCnt += 1;
                std::string u_addr = "0/";
                std::string v_addr = "0/";

                u_addr += std::to_string(recorder[colSrc]++);
                v_addr += std::to_string(recorder[colDst]++);

                u32 uDeviceId = row * 8 + colSrc;
                u32 vDeviceId = row * 8 + colDst;

                edge_list.push_back(json{{"net_layer", 0},
                    {"link_type", "PEER2PEER"},
                    {"protocols", {"UB_CTP"}},
                    {"local_a", uDeviceId},
                    {"local_a_ports", {u_addr}},
                    {"local_b", vDeviceId},
                    {"local_b_ports", {v_addr}},
                    {"position", "DEVICE"}});
                AddEidInfo(uDeviceId, u_addr, 0);
                AddEidInfo(vDeviceId, v_addr, 0);
                g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
                g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
            }
        }
    }

    if (!is1DTopo) {
        // 生成列方向，即die1上的链路
        for (u32 col = 0; col < 8; col++) {
            linkCnt = 0;
            vector<u32> recorder(8, 1);
            for (u32 rowSrc = 0; rowSrc < 8; rowSrc++) {
                u32 tmpDst = rowSrc + 1;
                for (u32 rowDst = rowSrc + 1; rowDst < 8; rowDst++) {
                    linkCnt += 1;
                    std::string u_addr = "1/";
                    std::string v_addr = "1/";

                    u_addr += std::to_string(recorder[rowSrc]++);
                    v_addr += std::to_string(recorder[rowDst]++);
                    u32 uDeviceId = rowSrc * 8 + col;
                    u32 vDeviceId = rowDst * 8 + col;

                    edge_list.push_back(json{{"net_layer", 0},
                        {"link_type", "PEER2PEER"},
                        {"protocols", {"UB_CTP"}},
                        {"local_a", uDeviceId},
                        {"local_a_ports", {u_addr}},
                        {"local_b", vDeviceId},
                        {"local_b_ports", {v_addr}},
                        {"position", "DEVICE"}});

                    AddEidInfo(uDeviceId, u_addr, 1);
                    AddEidInfo(vDeviceId, v_addr, 1);
                    g_uvDevice2Port[uDeviceId][vDeviceId] = u_addr;
                    g_uvDevice2Port[vDeviceId][uDeviceId] = v_addr;
                }
            }
        }
    }

    for (u32 row = 0; row < 8; row++) {
        for (u32 col = 0; col < 8; col++) {
            u32 uDeviceId = row * 8 + col;
            std::string u_addr = "0/7";
            edge_list.push_back(json{{"net_layer", 1},
                {"link_type", "PEER2NET"},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_addr}},
                {"position", "DEVICE"}});

            AddEidInfo(uDeviceId, u_addr, 0);
            
            std::string u_addr_die1 = "1/7";
            edge_list.push_back(json{{"net_layer", 1},
                {"link_type", "PEER2NET"},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_addr_die1}},
                {"position", "DEVICE"}});

            AddEidInfo(uDeviceId, u_addr_die1, 1);
        }
    }

    for (u32 row = 0; row < 8; row++) {
        for (u32 col = 0; col < 8; col++) {
            u32 uDeviceId = row * 8 + col;
            std::string u_addr = "0/8";
            edge_list.push_back(json{{"net_layer", 2},
                {"link_type", "PEER2NET"},
                {"protocols", {"UB_CTP"}},
                {"local_a", uDeviceId},
                {"local_a_ports", {u_addr}},
                {"position", "HOST"}});
            AddEidInfo(uDeviceId, u_addr, 1);
        }
    }

    topoJson["edge_count"] = edge_list.size();
    topoJson["edge_list"] = edge_list;

    const char *value = std::getenv("TOPO_PATH_NAME");
    if (value != nullptr) {
        std::string prefix(value);
        topoFileName = prefix + "_topo.json";
    } else {
        topoFileName = "topo.json";
    }

    std::ofstream topoFile(topoFileName);
    if (!topoFile.is_open()) {
        HCCL_ERROR("Failed to open file %s", topoFileName.c_str());
        return HcclResult::HCCL_E_PARA;
    }

    topoFile << topoJson.dump(4);
    topoFile.close();
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl