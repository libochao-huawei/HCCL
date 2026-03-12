/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: rank table stub header
 */

#ifndef HCCL_SIM_STUB_RANK_TABLE_H
#define HCCL_SIM_STUB_RANK_TABLE_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include "fwk_types.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

class SimNetworkManager {
public:
    explicit SimNetworkManager(DeviceType devType, std::pair<int, int> cluster, bool ccuFlag);
    SimNetworkManager()  = delete;
    ~SimNetworkManager() = default;

    bool RankTableFileCreate();
    void DelRankTableFile();
    void DelTopoInfoFile();
    void DelDieInfoFile();

public:
    enum TopoNetworkType {TOPO_TYPE_1D, TOPO_TYPE_2D, TOPO_TYPE_DETOUR, TOPO_TYPE_NUM, TOPO_TYPE_INVALID};

private:
    void GenTopoInfoXEdges(uint32_t row, uint32_t colSrc, uint32_t colDst, json &topoInfoJson);
    void GenTopoInfoYEdges(uint32_t rowSrc, uint32_t rowDst, uint32_t col, json &topoInfoJson);
    bool GenRankTableFileV1();
    bool Gen1DRankTableFileV2();
    bool Gen2DRankTableFileV2();
    bool GenDetourRankTableV2();

    bool GenTopoInfo(uint32_t rowMax, uint32_t colMax);
    bool GenDieInfo();

    TopoNetworkType GetTopoNetworkType();

private:
    static constexpr uint32_t SERVER_ROW_NUM = 8;
    static constexpr uint32_t SERVER_COL_NUM = 8;
    static constexpr uint32_t DIE_IP_GAP = 10;
    const char* fileRankTablePath = "ranktable.json";
    const char* fileTopoPath      = "topo.json";
    const char* fileDieInfoPath   = "die_info.json";

    DeviceType devType_{DeviceType::DEV_TYPE_NOSOC};
    int serverNum_{0};
    int devNum_{0};
    bool ccuFlag_{false};

    std::map<int, std::set<std::pair<std::string ,int>>> ip2DieInfo_{};
};
#endif