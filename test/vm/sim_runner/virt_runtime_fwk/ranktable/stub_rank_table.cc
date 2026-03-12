/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: rank table stub
 */

#include "stub_rank_table.h"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include "log.h"

SimNetworkManager::SimNetworkManager(DeviceType devType_, std::pair<int, int> cluster, bool ccuFlag)
    : devType_(devType_), serverNum_(cluster.first), devNum_(cluster.second), ccuFlag_(ccuFlag)
{}

// 坐标[i, j]到[i, m]的连接边，srcIp = 192.168.locIdj.locIdm ; dstIp = 192.168.locIdm,locIdj
void SimNetworkManager::GenTopoInfoXEdges(uint32_t row, uint32_t colSrc, uint32_t colDst, json &topoInfoJson)
{
    json xEdge;
    xEdge["level"] = 0;
    xEdge["protocol"] = "UB-CTP";

    uint32_t srcLocId = SERVER_COL_NUM * row + colSrc;
    uint32_t dstLocId = SERVER_COL_NUM * row + colDst;

    // Generate endpoint addresses
    json u_endpoint, v_endpoint;
    std::string srcIpAddr = "192.168." + std::to_string(srcLocId) +  "." + std::to_string(dstLocId);
    std::string dstIpAddr = "192.168." + std::to_string(dstLocId) +  "." + std::to_string(srcLocId);
    u_endpoint["type"] = "PEER";
    u_endpoint["id"] = srcLocId;
    u_endpoint["addr"] = srcIpAddr.c_str();
    u_endpoint["position"] = "device";

    v_endpoint["type"] = "PEER";
    v_endpoint["id"] = dstLocId;
    v_endpoint["addr"] = dstIpAddr.c_str();
    v_endpoint["position"] = "device";

    ip2DieInfo_[srcLocId].insert(std::make_pair(srcIpAddr, 0));   //  0为dieId
    ip2DieInfo_[dstLocId].insert(std::make_pair(dstIpAddr, 0));   //  0为dieId

    xEdge["u_endpoint"] = u_endpoint;
    xEdge["v_endpoint"] = v_endpoint;

    topoInfoJson["edge_list"].push_back(xEdge);
}

// 坐标[i, j]到[n, j]的连接边，srcIp = 192.168.i.n ; dstIp = 192.168.n.i
void SimNetworkManager::GenTopoInfoYEdges(uint32_t rowSrc, uint32_t rowDst, uint32_t col, json &topoInfoJson)
{
    json yEdge;
    yEdge["level"] = 0;
    yEdge["protocol"] = "UB-CTP";

    uint32_t srcLocId = SERVER_COL_NUM * rowSrc + col;
    uint32_t dstLocId = SERVER_COL_NUM * rowDst + col;

    // Generate endpoint addresses
    json u_endpoint, v_endpoint;
    std::string srcIpAddr = "192.168." + std::to_string(srcLocId) +  "." + std::to_string(dstLocId);
    std::string dstIpAddr = "192.168." + std::to_string(dstLocId) +  "." + std::to_string(srcLocId);
    u_endpoint["type"] = "PEER";
    u_endpoint["id"] = srcLocId;
    u_endpoint["addr"] = srcIpAddr.c_str();
    u_endpoint["position"] = "device";

    v_endpoint["type"] = "PEER";
    v_endpoint["id"] = dstLocId;
    v_endpoint["addr"] = dstIpAddr.c_str();
    v_endpoint["position"] = "device";

    ip2DieInfo_[srcLocId].insert(std::make_pair(srcIpAddr, 1));   //  1为dieId
    ip2DieInfo_[dstLocId].insert(std::make_pair(dstIpAddr, 1));   //  1为dieId

    yEdge["u_endpoint"] = u_endpoint;
    yEdge["v_endpoint"] = v_endpoint;

    topoInfoJson["edge_list"].push_back(yEdge);
}

bool SimNetworkManager::GenTopoInfo(uint32_t rowMax, uint32_t colMax)
{
    if (rowMax == colMax && rowMax == 1) {
        return true;
    }
    uint32_t devCountMax = rowMax * colMax;
    try {
        json topoInfoJson;
        topoInfoJson["version"] = "2.0";
        topoInfoJson["hardware_type"] = "910D-2D-Fullmsh_64_plus_1";
        // peer_list
        topoInfoJson["peer_count"] = devCountMax;
        topoInfoJson["peer_list"] = json::array();
        for (int i = 0; i < devCountMax; ++i) {
            json peer;
            peer["id"] = i;
            topoInfoJson["peer_list"].push_back(peer);
        }
        // dege_list
        uint32_t edgeCountX = ((colMax - 1) * colMax / 2) * rowMax;
        uint32_t edgeCountY = ((rowMax - 1) * rowMax / 2) * colMax;
        uint32_t edgeCount  = edgeCountX + edgeCountY;

        topoInfoJson["edge_count"] = edgeCount;
        topoInfoJson["edge_list"] = json::array();

        for (uint32_t row = 0; row < rowMax; ++row) {
            for (uint32_t colSrc = 0; colSrc < colMax - 1; ++colSrc) {
                for (uint32_t colDst = colSrc + 1; colDst < colMax; ++colDst) {
                    // X方向： die0 fullMesh
                    GenTopoInfoXEdges(row, colSrc, colDst, topoInfoJson);
                }
            }
        }

        for (uint32_t col = 0; col < colMax; ++col) {
            for (uint32_t rowSrc = 0; rowSrc < rowMax - 1; ++rowSrc) {
                for (uint32_t rowDst = rowSrc + 1; rowDst < rowMax; ++rowDst) {
                    // Y方向： die1 fullMesh
                    GenTopoInfoYEdges(rowSrc, rowDst, col, topoInfoJson);
                }
            }
        }

        // Write to file
        std::ofstream out(fileTopoPath, std::ofstream::out);
        out << topoInfoJson;  // Pretty print with 4 spaces indentation
    } catch (...) {
        HCCL_ERROR("[GenTopoInfo] Gen topo info file failed: file[%s]", fileTopoPath);
        return false;
    }
    return true;
}

// 根据server、device的个数自动生成ranktable.json
bool SimNetworkManager::GenRankTableFileV1()
{
    try {
        nlohmann::json rankTableJson;
        rankTableJson["server_count"] = std::to_string(serverNum_);
        rankTableJson["server_list"] = nlohmann::json();

        int deviceIndex = 0;
        for (int server = 0; server < serverNum_; ++server) {
            nlohmann::json serverObj = nlohmann::json::object();
            nlohmann::json deviceObj = nlohmann::json::array();

            for (int dev = 0; dev < devNum_; ++dev) {
                nlohmann::json device = nlohmann::json::object();
                device["device_id"] = std::to_string(deviceIndex);
                device["rank_id"] = std::to_string(deviceIndex);
                deviceObj.push_back(device);
                deviceIndex++; // 更新设备索引
            }

            serverObj["device"] = deviceObj;
            serverObj["server_id"] = std::to_string(server + 1);

            rankTableJson["server_list"].push_back(serverObj);
        }

        rankTableJson["status"] = "completed";
        rankTableJson["version"] = "1.0";
        std::ofstream  out(fileRankTablePath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        HCCL_ERROR("[GenRankTableFileV1] Gen rankTable file failed: file[%s]", fileRankTablePath);
        return false;
    }

    return true;
}

SimNetworkManager::TopoNetworkType SimNetworkManager::GetTopoNetworkType()
{
    if (!ccuFlag_) {
        return TopoNetworkType::TOPO_TYPE_INVALID;  // AICPU模式
    }

    // 判断CCU 2D流程
    const char *dieNum = std::getenv("HCCL_IODIE_NUM");
    if (dieNum != nullptr) {
        std::string value(dieNum);
        if (value == "2") {
            return TopoNetworkType::TOPO_TYPE_2D;
        }
    }

    // 判断CCU绕路流程
    const char *raoluEnv = std::getenv("HCCL_DETOUR");
    if (raoluEnv != nullptr) {
        std::string value(raoluEnv);
        if (value == "detour:1") {
            return TopoNetworkType::TOPO_TYPE_DETOUR;
        }
    }

    // CCU 1D流程
    return TopoNetworkType::TOPO_TYPE_1D;
}

/*
 * 单die算法：最大支持8P，默认按照rankSize，给出第0行的rankId
 */
bool SimNetworkManager::Gen1DRankTableFileV2()
{
    if (devNum_ > SERVER_ROW_NUM) {
        HCCL_ERROR("[Gen1DRankTableFileV2] 1D algorithm maximum support 8P, devNum[%d]", devNum_);
        return false;
    }
    try {
        nlohmann::ordered_json rankTableJson;
        rankTableJson["version"] = "2.0";
        rankTableJson["rank_count"] = std::to_string(devNum_);
        rankTableJson["rank_list"] = nlohmann::json();

        for (int i = 0; i < devNum_; ++i) {
            nlohmann::ordered_json rankObj;
            rankObj["rank_id"] = i;
            rankObj["local_id"] = i;

            nlohmann::ordered_json levelList;
            nlohmann::ordered_json levelObj;
            levelObj["level"] = 0;
            levelObj["id"] = "az0-rack0";
            levelObj["fabric_type"] = "INNER";
            levelObj["rank_addr_type"] = "";
            levelObj["rank_addrs"] = nlohmann::json::array();

            levelList.push_back(levelObj);
            rankObj["level_list"] = levelList;

            rankTableJson["rank_list"].push_back(rankObj);
        }

        std::ofstream out(fileRankTablePath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        HCCL_ERROR("[Gen1DRankTableFileV2] Gen rankTable file failed: file[%s]", fileRankTablePath);
        return false;
    }

    return true;
}

/*
 * 双die和绕路算法：用户指定rankId，组成mxn矩阵：m和n必须从2,4,8中选择
 * 默认双die算法的ranktable方案：4P 2x2方案：0 1 8 9
 * 8P 2x4方案： 0 1; 8 9; 16 17; 24 25
 * 16P 4x4方案：0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27
 * 32P 8x4方案：0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27; 32 33 34 35; 40 41 42 43; 48 49 50 51; 56 57 58 59
 * 64P 8x8方案：0~63
 */
bool SimNetworkManager::Gen2DRankTableFileV2()
{
    if (devNum_ != 4 && devNum_ != 8 && devNum_ != 16 && devNum_ != 32 && devNum_ != 64) {
        HCCL_ERROR("[Gen2DRankTableFileV2] The dual-die algorithm provides only 4P, 8P, 16P, 32P and 64P ranktable files by default, devNum[%d]", devNum_);
        return false;
    }
    try {
        nlohmann::ordered_json rankTableJson;
        rankTableJson["version"] = "2.0";
        rankTableJson["rank_count"] = std::to_string(devNum_);
        rankTableJson["rank_list"] = nlohmann::json();

        for (int i = 0; i < devNum_; ++i) {
            nlohmann::ordered_json rankObj;
            rankObj["rank_id"] = i;
            if (devNum_ == 4 || devNum_ == 8) { // 2x2、2x4方案
                rankObj["local_id"] = 8 * (i / 2) + i % 2;
            } else if (devNum_ == 16 || devNum_ == 32) { // 4x4、4x8方案
                rankObj["local_id"] = 8 * (i / 4) + i % 4;
            } else { // 8x8方案
                rankObj["local_id"] = 8 * (i / 8) + i % 8;
            }

            nlohmann::ordered_json levelList;
            nlohmann::ordered_json levelObj;
            levelObj["level"] = 0;
            levelObj["id"] = "az0-rack0";
            levelObj["fabric_type"] = "INNER";
            levelObj["rank_addr_type"] = "";
            levelObj["rank_addrs"] = nlohmann::json::array();

            levelList.push_back(levelObj);
            rankObj["level_list"] = levelList;

            rankTableJson["rank_list"].push_back(rankObj);
        }

        std::ofstream out(fileRankTablePath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        HCCL_ERROR("[Gen2DRankTableFileV2] Gen rankTable file failed: file[%s]", fileRankTablePath);
        return false;
    }

    return true;
}

/*
 * 双die和绕路算法：用户指定rankId，组成mxn矩阵：m和n必须从2,4,8中选择
 * 默认绕路算法的ranktable方案：【注意】当前绕路只支持4P场景
 * 4P 2x2方案：0 1
 * 8P 1x8方案： 0 1 2 3;
 * 16P 4x4方案：0 1 2 3; 8 9 10 11;
 * 32P 4x8方案：0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27;
 * 64P 8x8方案：0~31
 */
bool SimNetworkManager::GenDetourRankTableV2()
{
    if (devNum_ != 2) {
        HCCL_ERROR("[GenDetourRankTableV2] Only support detour:1 for now. devNum=[%d]", devNum_);
        return false;
    }
    try {
        nlohmann::ordered_json rankTableJson;
        rankTableJson["version"] = "2.0";
        rankTableJson["rank_count"] = std::to_string(devNum_);
        rankTableJson["rank_list"] = nlohmann::json();

        for (int i = 0; i < devNum_; ++i) {
            nlohmann::ordered_json rankObj;
            rankObj["rank_id"] = i;
            // 当前绕路只支持4P场景
            rankObj["local_id"] = i;

            nlohmann::ordered_json levelList;
            nlohmann::ordered_json levelObj;
            levelObj["level"] = 0;
            levelObj["id"] = "az0-rack0";
            levelObj["fabric_type"] = "INNER";
            levelObj["rank_addr_type"] = "";
            levelObj["rank_addrs"] = nlohmann::json::array();

            levelList.push_back(levelObj);
            rankObj["level_list"] = levelList;

            rankTableJson["rank_list"].push_back(rankObj);
        }

        std::ofstream out(fileRankTablePath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        HCCL_ERROR("[GenDetourRankTableV2] Gen rankTable file failed: file[%s]", fileRankTablePath);
        return false;
    }

    return true;
}

bool SimNetworkManager::GenDieInfo()
{
    int devCountMax = SERVER_ROW_NUM * SERVER_COL_NUM;
    try {
        json dieInfoJson;
        dieInfoJson["version"] = "2.0";
        dieInfoJson["hardware_type"] = "910D-2D-Fullmsh_64_plus_1";
        dieInfoJson["edge_count"] = devCountMax;
        dieInfoJson["edge_list"] = json::array();
        // Generate edge list
        int count = 0;
        for (auto it = ip2DieInfo_.begin(); it != ip2DieInfo_.end(); ++it) {
            json rankInfo;
            rankInfo["local_id"] = it->first;
            rankInfo["die_info"] = json::array();
            for (auto &ipInfo : it->second) {
                json dieInfo;
                dieInfo["die_id"] = ipInfo.second;
                dieInfo["ip"] = ipInfo.first;
                rankInfo["die_info"].push_back(dieInfo);
            }
            dieInfoJson["edge_list"].push_back(rankInfo);
        }

        // Write to file
        std::ofstream out(fileDieInfoPath, std::ofstream::out);
        out << dieInfoJson;  // Pretty print with 4 spaces indentation
    } catch (...) {
        HCCL_ERROR("[GenDieInfo] Gen die info file failed: file[%s]", fileDieInfoPath);
        return false;
    }
    return true;
}

bool SimNetworkManager::RankTableFileCreate()
{
    // 目前只支持单机1-8卡
    if (serverNum_ != 1) {
        HCCL_ERROR("[RankTableFileCreate] only support 1 server now.");
        return false;
    }

    if (devType_ == DeviceType::DEV_TYPE_V80 && devNum_ > 8) {
        HCCL_ERROR("[RankTableFileCreate] [%s] only support 1-8 devices now.", devType_.Describe().c_str());
        return false;
    }

    if ((devType_ == DeviceType::DEV_TYPE_V71 || devType_ == DeviceType::DEV_TYPE_910_93) && devNum_ > 16) {
        HCCL_ERROR("[RankTableFileCreate] [%s] only support 1-16 devices now.", devType_.Describe().c_str());
        return false;
    }

    // 910D David的ranktable.json是2.0版本的
    bool genRankTableFileFalg = true;
    bool genTopoInfoFalg = true;
    if (devType_ == DeviceType::DEV_TYPE_910_95) {
        auto topoNetworkType = GetTopoNetworkType();
        if (topoNetworkType == TopoNetworkType::TOPO_TYPE_1D) {
            genRankTableFileFalg = Gen1DRankTableFileV2();
            genTopoInfoFalg = GenTopoInfo(1, SERVER_COL_NUM);
        } else if (topoNetworkType == TopoNetworkType::TOPO_TYPE_2D) {
            genRankTableFileFalg = Gen2DRankTableFileV2();
            if (devNum_ == 4) { // 4P 2x2方案：0 1 8 9
                genTopoInfoFalg = GenTopoInfo(2, 2);
            } else if (devNum_ == 8) { // 8P 2x4方案： 0 1; 8 9; 16 17; 24 25
                genTopoInfoFalg = GenTopoInfo(2, 4);
            } else if (devNum_ == 16) { // 16P 4x4方案：0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27
                genTopoInfoFalg = GenTopoInfo(4, 4);
            } else if (devNum_ == 32) { // 32P 8x4方案：0 1 2 3; 8 9 10 11; 16 17 18 19; 24 25 26 27; 32 33 34 35; 40 41 42 43; 48 49 50 51; 56 57 58 59
                genTopoInfoFalg = GenTopoInfo(SERVER_ROW_NUM, 4);
            } else if (devNum_ == 64) { // 64P 8x8方案：0~63
                genTopoInfoFalg = GenTopoInfo(SERVER_ROW_NUM, SERVER_COL_NUM);
            } else {
                HCCL_ERROR("[RankTableFileCreate] Not support device num[%u] for 2D algorithm.", devNum_);
                return false;
            }
        } else if (topoNetworkType == TopoNetworkType::TOPO_TYPE_DETOUR) {
            // 绕路场景：业务目前仅支持1D绕路 ———— topoInfo.json文件中，只生成X方向的边
            genRankTableFileFalg = GenDetourRankTableV2();
            genTopoInfoFalg = GenTopoInfo(1, SERVER_COL_NUM);
        } else {
            // 非CCU模式(David AICPU模式)
            genRankTableFileFalg = Gen1DRankTableFileV2();
            genTopoInfoFalg = GenTopoInfo(1, SERVER_COL_NUM);
        }

        bool genDieInfoFalg = GenDieInfo();
        return genRankTableFileFalg && genTopoInfoFalg && genDieInfoFalg;
    }

    return GenRankTableFileV1();
}

void SimNetworkManager::DelRankTableFile()
{
    int res = unlink(fileRankTablePath);
    if (res == -1) {
        HCCL_ERROR("[DelRankTableFile] delete rankTable file failed: file[%s]", fileRankTablePath);
        return;
    }
}

void SimNetworkManager::DelTopoInfoFile()
{
    int res = unlink(fileTopoPath);
    if (res == -1) {
        HCCL_ERROR("[DelRankTableFile] delete topo info file failed: file[%s]", fileTopoPath);
        return;
    }
}

void SimNetworkManager::DelDieInfoFile()
{
    int res = unlink(fileDieInfoPath);
    if (res == -1) {
        HCCL_ERROR("[DelRankTableFile] delete die info file failed: file[%s]", fileDieInfoPath);
        return;
    }
}