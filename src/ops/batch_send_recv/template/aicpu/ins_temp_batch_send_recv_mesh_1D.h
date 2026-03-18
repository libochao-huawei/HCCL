/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INS_TEMP_BATCH_SEND_RECV_MESH_1D_H
#define INS_TEMP_BATCH_SEND_RECV_MESH_1D_H

#include <deque>
#include <set>
#include "alg_v2_template_base.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"

namespace ops_hccl {

class InsTempBatchSendRecvMesh1D : public InsAlgTemplateBase {
public:
    explicit InsTempBatchSendRecvMesh1D(const OpParam &param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempBatchSendRecvMesh1D() override = default;

    std::string Describe() const override
    {
        return "Template of BatchSendRecv Mesh1D";
    }

    HcclResult KernelRun(const OpParam &param,
        const TemplateDataParams &tempAlgParams,
        const TemplateResource &templateResource) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

    void SetBatchSendRecvInfo(const BatchSendRecvInfo &info);

private:
    static constexpr u32 CHANNEL_NUM_PER_RANK_PAIR = 2;

    HcclResult ProcessSelfSendRecvTasks(const ThreadHandle &thread);
    HcclResult GetSendChannel(u32 remoteRank,
        const std::map<u32, std::vector<ChannelInfo>> &channels, ChannelInfo &sendChannel) const;
    HcclResult GetRecvChannel(u32 remoteRank,
        const std::map<u32, std::vector<ChannelInfo>> &channels, ChannelInfo &recvChannel) const;
    HcclResult ProcessSendDataSlice(SendRecvSlice &sendSlice, const ThreadHandle &thread,
        const HcclMem &cclMem, const std::map<u32, std::vector<ChannelInfo>> &channels) const;
    HcclResult ProcessRecvDataSlice(SendRecvSlice &recvSlice, const ThreadHandle &thread,
        const std::map<u32, std::vector<ChannelInfo>> &channels) const;
    HcclResult RunLoopSendRecv(const std::vector<ThreadHandle> &threads, const HcclMem &cclMem,
        const std::map<u32, std::vector<ChannelInfo>> &channels);

    BatchSendRecvInfo batchSendRecvInfo_;
};

} // namespace ops_hccl
#endif // INS_TEMP_BATCH_SEND_RECV_MESH_1D_H
