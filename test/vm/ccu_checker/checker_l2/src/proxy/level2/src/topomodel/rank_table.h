#ifndef SIM_RANK_TABLE_H
#define SIM_RANK_TABLE_H

#include "hccl_proxy_pub.h"
#include "hccl_common.h"
#include "dtype_common.h"
#include "hccl_rankgraph.h"
#include "hccl_common_defs.h"
#include "hccl_sim_data_defs.h"

namespace HcclProxy {

HcclResult GenGraphRankInfos(ShmCommDomain* commDomain, std::vector<GraphRankInfo> &rankGraphs);

};

#endif  // SIM_RANK_TABLE_H