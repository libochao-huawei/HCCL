#ifndef ADAPTER_V2_COLL_GEN_JSON_H
#define ADAPTER_V2_COLL_GEN_JSON_H

#include <map>
#include <vector>

#include "hccl_types.h"
#include "topo_meta.h"
#include "log.h"
#include "orion_adapter_hccp.h"
#include "checker_def.h"
using namespace checker;

namespace Hccl {
extern std::map<u32, std::vector<HrtDevEidInfo>> g_devId2EidInfo;
extern std::map<u32, std::map<IpAddress, std::pair<uint32_t, uint32_t>>> g_devId2Ip2DieIdAndFuncId;
extern std::map<u32, u32> g_devId2funcId;
extern std::map<u32, map<std::string, u32>> g_devId2PortId2DieId;
extern std::map<u32, map<std::string, u32>> g_devId2PortId2funcId;
extern std::map<u32, map<std::string, HrtDevEidInfo>> g_devId2PortId2EidInfo;

HcclResult InitGenRankTableJson(TopoMeta& topoMeta, const CheckerOpParam& param, std::string& rankTableString);
HcclResult InitGenRankTableJsonHF(TopoMeta& topoMeta, std::string& rankTableString);
HcclResult InitGenTopoJson(std::string &topoFileName, const CheckerOpParam& param, bool is1DTopo);
HcclResult InitGenTopoJsonHF(std::string &topoFileName, bool is1DTopo);
}

#endif
