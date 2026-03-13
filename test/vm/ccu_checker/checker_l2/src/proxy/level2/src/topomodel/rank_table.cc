#include "topo_model.h"
#include "rank_table.h"

namespace HcclProxy {
HcclResult GenGraphRankInfos(ShmCommDomain* commDomain, std::vector<GraphRankInfo> &rankGraphs)
{
    u32 superPodId = 0;
    u32 serverId = 0;
    u32 rankId = 0;
    u32 boxIpStart = 168430090;   // Server起始IP
    u32 devIpStart = 3232238090;  // 设备起始IP
    for (size_t i = 0; i < commDomain->rankNum; ++i) {
        superPodId = (commDomain->rankId2NpuPos)[i].field.podId;
        serverId = (commDomain->rankId2NpuPos)[i].field.serId;
        u32 phyDeviceId = (commDomain->rankId2NpuPos)[i].field.phyId;

        CommAddr hostIp;
        hostIp.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
        u32 npuBoxIp = boxIpStart + serverId; // 暂定只有一个pod，多pod要考虑serid重复情况
        hostIp.id = htonl(npuBoxIp);

        GraphRankInfo rankGraph;
        rankGraph.rankId = i;
        rankGraph.serverIdx = serverId;
        rankGraph.serverId = std::to_string(serverId);
        rankGraph.superDeviceId = superPodId;
        rankGraph.superPodId = std::to_string(superPodId);
        rankGraph.hostIp = hostIp;

        CommAddr devIp;
        devIp.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
        u32 npuDevIp = devIpStart + i;
        devIp.id = htonl(npuDevIp);
        rankGraph.deviceInfo.deviceIp = devIp;
        rankGraph.deviceInfo.devicePhyId = phyDeviceId;
        rankGraphs.push_back(rankGraph);
    }
    return HCCL_SUCCESS;
}

};