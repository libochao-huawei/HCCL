/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 适配1.0生成task队列的函数
 * Author: huangweihao
 * Create: 2025-05-06
 */

#ifndef HCCL_ADAPTER_V1_ORCHESTRATE_H
#define HCCL_ADAPTER_V1_ORCHESTRATE_H

#include <map>
#include <unordered_map>
#include <vector>

#include "transformer.h"
#include "hccl_types.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "transport.h"
#include "comm.h"
#include "communicator_stub.h"

using namespace checker;
namespace hccl {

extern std::unordered_map<RankId, std::vector<std::shared_ptr<TransportCompared>>> AllTransport_;
extern std::unordered_map<Transport*, std::shared_ptr<TransportCompared>> links2TransportCompare_;
extern map<TransportType, unordered_map<RankId, unordered_map<RankId, std::shared_ptr<Transport>>>> CreatedLinksDict_;
HcclResult CheckTransportLink();
HcclResult InitCommParams(HcclCommParams &params, RankTable_t& rankTable, RankId myRank);
void InitOpParam(OpParam &opParam, CheckerOpParam &checkerOpParam, RankId myRank,
    u32 rankSize, bool initStream, bool isIOSameAddr);
HcclResult GenRankTable(hccl::RankTable_t &rankTable, TopoMeta topoMate);
HcclResult OrchestraTask(CheckerOpParam &checkerOpParam, RankTable_t &rankTable, u32 rankNum, bool isRunning,
    std::vector<std::shared_ptr<hccl::HcclCommunicator>> &communicators, bool isIOSameAddr);

} // namespace checker

#endif