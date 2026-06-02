/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_MESH_1D_H
#define INS_TEMP_ALL_TO_ALL_V_MESH_1D_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

const uint32_t ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE = 16; // fullmesh最大的并发数量

class InsTempAlltoAllVMesh1D : public InsAlgTemplateBase {
public:
    InsTempAlltoAllVMesh1D() = default;
    explicit InsTempAlltoAllVMesh1D(const OpParam& param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempAlltoAllVMesh1D() override;

    std::string Describe() const override
    {
        std::string info = "Template of alltoallv Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    // 现在的RunAsync就是之前的GenExtIns
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         TemplateResource& templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    struct MatrixAlltoAllSlot {
        u32 txRank{INVALID_VALUE_RANKID};
        u32 rxRank{INVALID_VALUE_RANKID};
        u32 channelIdx{0};
        u32 cclBuffIdx{0};
        bool isMesh{false};
    };

    bool IsMatrixAlltoAllRankSize(u32 &matrixDim) const;
    bool IsMatrixAlltoAllParam(const OpParam &param) const;
    bool IsMatrixAlltoAllTopo(const TopoInfoWithNetLayerDetails *topoInfo) const;
    u32 CalcMatrixRank(const u32 row, const u32 col, const u32 matrixDim) const;
    HcclResult CalcMatrixAlltoAllRoundPlan(const u32 round, const u32 myAlgRank,
        std::vector<MatrixAlltoAllSlot> &slotPlans) const;
    HcclResult CheckMatrixAlltoAllChannels(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const u32 matrixDim, const u32 myAlgRank) const;
    HcclResult SelectMatrixAlltoAllChannel(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const MatrixAlltoAllSlot &slotPlan, const bool selectTxChannel, ChannelInfo &channel) const;
    HcclResult RunMatrixAlltoAll(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams, const u32 myAlgRank);
    HcclResult RunMatrixAlltoAllReadPipelined(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams, const u32 myAlgRank,
        const u32 matrixDim);
    HcclResult RunMatrixAlltoAllPreCopyRound(const TemplateDataParams &tempAlgParams,
        const std::vector<ThreadHandle> &copyThreads, const u32 round, const u32 myAlgRank, const u32 bank) const;
    HcclResult RunMatrixAlltoAllCommRound(const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &commThreads,
        const u32 round, const u32 myAlgRank, const u32 bank, const bool needPreCopy) const;
    HcclResult RunMatrixAlltoAllSlot(const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const MatrixAlltoAllSlot &slotPlan,
        const ThreadHandle &thread, const u32 round, const bool needPreCopy) const;

    HcclResult RunALLtoALL(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams, const u32 myAlgRank);
    HcclResult PreCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
        const u32 myRankCclBuffIdx, const u32 remoteRank, const u64 &sendSize,
        const u64 &sendCount, const u64 &sendOffset) const;
    HcclResult PreCopyByLoop(const std::vector<u32> &commRanks,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams, const u32 myAlgRank);
    HcclResult PostCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
        const u32 myRankCclBuffIdx, const u32 remoteRank, const u64 &recvSize,
        const u64 &recvCount, const u64 &recvOffset) const;
    HcclResult LocalCopyForMyRank(const TemplateDataParams &tempAlgParams,
        const ThreadHandle &thread, const u32 myAlgRank, const u32 queIdx) const;
    void CalcCommRankSetForOneLoop(const u32 roundIdx, const u32 remainRankSize, std::vector<u32> &commRanks) const;
    u32 CalcCommLoops() const;
    void CalcCclBuffIdx(u32 remoteRank, u32 &myRankCclBuffIdx, u32 &remoteCclBuffIdx) const;
    HcclResult RunSendRecvByLoop(const std::vector<u32> &commRanks, const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const u32 roundIdx, const u32 commLoops);
    HcclResult RunSendRecvByChannel(const TemplateDataParams &tempAlgParams, const u32 roundIdx,
        const u32 curValidChannelsSize, const std::vector<ChannelInfo> &curChannels, const u32 remoteRank,
        const std::vector<ThreadHandle> &threads, const u32 commLoops) const;
    HcclResult RunSendRecv(const TemplateDataParams &tempAlgParams,
        const SendRecvInfo &sendRecvInfo, const DataInfo &sendInfo, const DataInfo &recvInfo,
        const ThreadHandle& thread, const u32 channelId) const;
    HcclResult PreSyncInterThreadsPerRank(const ThreadHandle &mainThreadCurRank,
        const std::vector<ThreadHandle> &subThreadsCurRank) const;
    HcclResult PostSyncInterThreadsPerRank(const ThreadHandle &mainThreadCurRank,
        const std::vector<ThreadHandle> &subThreadsCurRank) const;

    u64 dataTypeSize_{0};
    bool isDmaRead_{false};
    u32 concurrentSendRecvNum_{1};
    std::vector<u64> sendCountsSplit_;
    std::vector<u64> sendSizeSplit_;
    std::vector<u64> sendOffsetSplit_;
    std::vector<u64> recvCountsSplit_;
    std::vector<u64> recvSizeSplit_;
    std::vector<u64> recvOffsetSplit_;
};

} // namespace Hccl

#endif //INS_TEMP_ALL_TO_ALL_V_MESH_1D_H
