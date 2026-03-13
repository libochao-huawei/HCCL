/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_STUB_COMMUNICATOR_H
#define HCCL_STUB_COMMUNICATOR_H

#include <atomic>
#include <memory>
#include <hccl/hccl_types.h>
#include <string>

#include "ccl_buffer_manager.h"

#include "hccl_communicator_attrs.h"
#include "hccl/base.h"
#include "hccl_alg.h"
#include "comm.h"
#include "device_capacity.h"
#include "coll_alg_operator.h"
#include "alltoall_operator.h"
#include "coll_alg_utils.h"
#include "profiler_manager.h"
using namespace std;

namespace hccl {
extern std::string g_algName;

class HcclCommunicator {
public:
    explicit HcclCommunicator();
    virtual ~HcclCommunicator();
    virtual HcclResult Init(HcclCommParams &params, const RankTable_t &rankTable);
    HcclResult ExecOp(HcclCMDType opType, OpParam &opParam, bool isRunning, string givenAlgName, u32 aiCoreLimit);
    HcclResult SetAlgOpContext(AlgOpContext algOpContext);
    HcclResult GetAivTag(std::string algName, bool isCapture, s32 &aivTag);

    std::unordered_map<std::string, AlgResourceResponse> resMap_; // tag : AlgResourceResponse

private:
    HcclResult CreateNotifies(u32 notifyNum, vector<shared_ptr<LocalNotify>> &NotifysM2S,
        vector<shared_ptr<LocalNotify>> &NotifysS2M);
    HcclResult CreateTransport(OpCommTransport &algResRequest, RankId rankId, OpCommTransport &algRespond, const bool &isZeroCopy, const HcclCMDType &opType);
    HcclResult CreateStream(u32 streamNum, vector<Stream>& streams);
    HcclResult RefreshMemLayoutAndGetMemResponse(const OpParam &opParam, AlgResourceRequest &resRequest,
        AlgResourceResponse &algResResponse, RankId rankId);
    LinkType GetLinkType(TransportType transportType, u32 localRank, u32 remoteRank);
    HcclResult InitCommParams(HcclCommParams &params);
    HcclResult InitRankInfo(const RankTable_t &rankTable);
    HcclResult InitDispatcher();
    HcclResult InitPara();
    HcclResult AllocAlgResource(const std::string &newTag, const OpParam &opParam,
        AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse);
    void GenAllGatherResultForAllToAllV(OpParam &opParam, void* result);

    HcclAlg *implAlg_ = nullptr;
    HcclCommunicatorAttrs attrCollector_;

    HcclDispatcher dispatcher_; // dispatcher放到最后析构
    HcclDispatcher vDispatcher_; // virtualDispatcher放到最后析构
    CCLBufferManager cclBufferManager_;

    u32 userRank_;  // 本group中的userrank
    u32 realUserRank_;  // world group中的userrank
    u32 userRankSize_;
	s32 deviceLogicId_;
	bool hcomGroupNicInit_;
	std::string profilingOption_;
	DevType deviceType_;
    std::string collectiveId_;
    HcclComm commHandle_;
	WorkMode commWorkMode_;
    u32 meshAggregationRankSize_;
	std::string identifier_;
    u32 ranktableCrc_;
	bool profilingInitiated_;
	HcclCommConnections commConnections_;

    u32 devicePhyId_;
    std::shared_ptr<ProfilerManager> profilerManager_;
    AlgOpContext algOpContext_;

    s32 aivOpbaseTag_ = 1;
    s32 aivOffloadTag_ = 1;
};

}  // end namespace hccl
#endif  // HCCL_IMPL_BASE_H
