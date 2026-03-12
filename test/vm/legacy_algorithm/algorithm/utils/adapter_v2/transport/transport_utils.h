#ifndef HCCL_ADAPTERV2_TRANSPORT_UTILS_H
#define HCCL_ADAPTERV2_TRANSPORT_UTILS_H

#include <map>
#include <vector>
#include "llt_common.h"
#include "ccu_transport.h"

using namespace checker;
namespace Hccl {

struct RemoteDieInfo {
    RankId dstRank;
    uint32_t remoteDieId;
};

using ChannelsPerDie = std::map<uint32_t, RemoteDieInfo>;

// rank <-> dieId <-> channelId <-> dieInfo
extern std::map<RankId, std::map<u32, ChannelsPerDie>> g_allRankChannelInfo;

// localRank <-> remoteRank <-> transports
// 绕路场景下，两个rank之间会有多个多条链路，此时需要根据localAddr和remoteAddr进行区分
extern std::map<RankId, std::map<RankId, std::vector<CcuTransport*>>> g_allRankTransports;

extern std::map<CcuTransport*, CcuTransport*> g_transportsPair;

}

#endif