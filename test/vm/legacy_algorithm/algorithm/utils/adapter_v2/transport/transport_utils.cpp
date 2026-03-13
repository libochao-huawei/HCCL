
#include "transport_utils.h"
#include "ccu_transport.h"

namespace Hccl {

std::map<RankId, std::map<u32, ChannelsPerDie>> g_allRankChannelInfo;
std::map<RankId, std::map<RankId, std::vector<CcuTransport*>>> g_allRankTransports;
std::map<CcuTransport*, CcuTransport*> g_transportsPair;

}