/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H
#define HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H

#include <algorithm>
#include <deque>
#include <set>

#include "executor_common_ops.h"
#include "topo_match_1d.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2BatchSendRecvSoleExecutor : public InsCollAlgBase {
public:
    explicit InsV2BatchSendRecvSoleExecutor();
    ~InsV2BatchSendRecvSoleExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm,
        TopoInfoWithNetLayerDetails *topoInfo,
        AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

private:
    HcclResult ParseAndOrganize(const HcclSendRecvItem *itemPtr, u32 itemNum);
    HcclResult GetPairWiseList(const HcclSendRecvItem *sendRecvInfo, u32 itemNum);
    HcclResult CalcSendSlices();
    HcclResult CalcRecvSlices();
    HcclResult CalcSelfSlices();

    bool SortSendItems(const HcclSendRecvItem *a, const HcclSendRecvItem *b) const;
    bool SortRecvItems(const HcclSendRecvItem *a, const HcclSendRecvItem *b) const;
    bool SortSelfItems(const HcclSendRecvItem *a, const HcclSendRecvItem *b) const;

    const HcclSendRecvItem *itemPtr_ = nullptr;
    u32 itemNum_ = 0;
    std::deque<const HcclSendRecvItem *> sendDeque_;
    std::deque<const HcclSendRecvItem *> recvDeque_;
    std::deque<const HcclSendRecvItem *> sendToSelfDeque_;
    std::deque<const HcclSendRecvItem *> recvFromSelfDeque_;
    std::deque<SendRecvSlice> sendDataSlices_;
    std::deque<SendRecvSlice> recvDataSlices_;
    std::deque<SendRecvSlice> selfSendSlices_;
    std::deque<SendRecvSlice> selfRecvSlices_;
};

} // namespace ops_hccl
#endif // HCCLV2_INS_V2_BATCH_SEND_RECV_EXECUTOR_H
