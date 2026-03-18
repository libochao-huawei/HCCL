/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_BATCH_SEND_RECV_MESH_1D_H
#define HCCL_CCU_TEMP_BATCH_SEND_RECV_MESH_1D_H

#include <map>
#include <vector>
#include "ccu_alg_template_base.h"

namespace ops_hccl {

struct PeerSendInfo {
    uint64_t sendAddr;
    uint64_t sendLen;
};

struct PeerItemInfo {
    uint64_t sendAddr;
    uint64_t sendLen;
    uint64_t recvAddr;
};

class CcuTempBatchSendRecvMesh1D : public CcuAlgTemplateBase {
public:
    explicit CcuTempBatchSendRecvMesh1D(const OpParam &param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempBatchSendRecvMesh1D() override = default;

    std::string Describe() const override
    {
        return "Template of BatchSendRecv CCU Mesh1D";
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param,
        const TemplateDataParams &templateDataParams,
        const TemplateResource &templateResource) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void SetBatchSendRecvInfo(const BatchSendRecvInfo &info);

private:
    HcclResult ParseTasks();

    BatchSendRecvInfo batchSendRecvInfo_;
    std::vector<u32> actualPeers_;                          // 实际参与通信的 rank 列表（含 self），升序
    std::map<u32, std::vector<PeerItemInfo>> peerItems_;    // peer -> item 列表
    u32 maxRound_ = 0;
    uint32_t mySubCommRank_ = 0;
};

} // namespace ops_hccl
#endif // HCCL_CCU_TEMP_BATCH_SEND_RECV_MESH_1D_H
