/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_model.h"

using namespace hccl;

namespace HcclSim {

int AsyParamTopo(TopoMeta &topoMeta)
{
    return 0;
}

HcclResult GenTopoMeta(TopoMeta &topoMate, int superPodNum, int serverNum, int rankNum)
{
    for (u32 i = 0; i < superPodNum; i++) {  // box
        SuperPodMeta superPodMeta;
        for (u32 j = 0; j < serverNum; j++) {  // serverNumPerBox
            ServerMeta serverMate;
            for (u32 k = 0; k < rankNum; k++) {
                serverMate.push_back(k);
            }
            superPodMeta.push_back(serverMate);
        }
        topoMate.push_back(superPodMeta);
    }
    return HCCL_SUCCESS;
}

HcclResult GenRankTable(TopoMeta topoMate, RankTable_t &rankTable)
{
    u32 currentRankId = 0;        // rankNum 及 rankid 计数器
    u32 boxIpStart = 168430090;   // 超节点 起始IP(主机序)
    u32 devIpStart = 3232238090;  // 设备 起始 IP (主机序)
    u32 superPodId_ = 0;          // 超级节点 superPodId
    u32 serverIdx_ = 0;           // 服务器下标编号
    // Box 遍历
    std::vector<SuperPodMeta>::iterator topomate_item_begin = topoMate.begin();
    std::vector<SuperPodMeta>::iterator topomate_item_end = topoMate.end();
    for (; topomate_item_begin != topomate_item_end; topomate_item_begin++) {
        u32 superDeviceId_in_pod = 0;  // 超节点内 superDeviceId 计数器

        // Server 遍历
        std::vector<ServerMeta>::iterator ServerMate_item_begin = (*topomate_item_begin).begin();
        std::vector<ServerMeta>::iterator ServerMate_item_end = (*topomate_item_begin).end();
        for (; ServerMate_item_begin != ServerMate_item_end; ServerMate_item_begin++) {
            HcclIpAddress newServerIp(htonl(boxIpStart++));  // 当前服务器的的 IP 指派
            // device 遍历
            std::vector<PhyDeviceId>::iterator device_item_begin = (*ServerMate_item_begin).begin();
            std::vector<PhyDeviceId>::iterator device_item_end = (*ServerMate_item_begin).end();
            for (; device_item_begin != device_item_end; device_item_begin++) {
                // *device_item_begin 是每个 devicePhdId
                HcclIpAddress newDeviceIp(htonl(devIpStart++));             // 当前 device 的 IP 指派
                RankInfo_t temp_rankInfo;                                   // 临时 rankInfo
                temp_rankInfo.rankId = currentRankId++;                     // rankId 自增 1
                temp_rankInfo.hostIp = newServerIp;                         // serverIp 对象
                temp_rankInfo.serverId = newServerIp.GetReadableIP();       // serverIP 的点分十进制
                temp_rankInfo.superPodId = std::to_string(superPodId_);     // 超节点ID
                temp_rankInfo.superPodIdx = superPodId_;                    // 超节点ID
                temp_rankInfo.superDeviceId = superDeviceId_in_pod++;       // 超节点内  deviceID
                temp_rankInfo.deviceInfo.devicePhyId = *device_item_begin;  // 服务器内  device 标识
                temp_rankInfo.deviceInfo.deviceIp.push_back(newDeviceIp);   // 服务器内  deviceIP
                temp_rankInfo.serverIdx = serverIdx_;                       // 服务器下标编号
                rankTable.rankList.push_back(temp_rankInfo);                // 将临时 rankInfo 插入到rankTable 中
            }
            devIpStart += 256;  // 更新下一个服务器(刀片)下的设备起始 IP
            serverIdx_++;
            ServerInfo_t temp_serverInfo;
            temp_serverInfo.serverId = newServerIp.GetReadableIP();  // serverId
            rankTable.serverList.push_back(temp_serverInfo);         // 将临时temp_serverInfo 插入 serverList
        }
        superPodId_++;
        boxIpStart += 256;  // 更新下一个 SupePod 超级节点的起始 IP
    }
    rankTable.serverNum = rankTable.serverList.size();  // 刀片数 server 数
    rankTable.deviceNum = rankTable.rankList.size();    // device 总数
    rankTable.rankNum =
        (currentRankId == rankTable.rankList.size()) ? rankTable.rankList.size() : currentRankId;  // rank 总数
    rankTable.superPodNum = topoMate.size();                     // 超级节点数 superPodNum
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;  // Device 网卡挂载位置

    return HCCL_SUCCESS;
}

};