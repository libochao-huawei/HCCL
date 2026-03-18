/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "kernel/ccu_kernel_batch_send_recv_mesh1d.h"
#include "ccu_temp_batch_send_recv_mesh_1D.h"

namespace ops_hccl {

CcuTempBatchSendRecvMesh1D::CcuTempBatchSendRecvMesh1D(const OpParam &param,
    const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

HcclResult CcuTempBatchSendRecvMesh1D::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);

    // 收集实际参与通信的 peer ranks
    CHK_PTR_NULL(param.batchSendRecvDataDes.sendRecvItemsPtr);
    const HcclSendRecvItem *itemPtr = param.batchSendRecvDataDes.sendRecvItemsPtr;
    u32 itemNum = param.batchSendRecvDataDes.itemNum;

    std::set<u32> actualPeerSet;
    for (u32 i = 0; i < itemNum; i++) {
        actualPeerSet.insert((itemPtr + i)->remoteRank);
    }
    std::vector<u32> actualPeers(actualPeerSet.begin(), actualPeerSet.end());

    // 为每个非 self 的 peer 申请 1 条 channel
    std::vector<HcclChannelDesc> channelDescs;
    for (const u32 &peer : actualPeers) {
        if (peer == static_cast<u32>(topoInfo->userRank)) {
            continue;
        }
        std::vector<HcclChannelDesc> channelByRank;
        CHK_RET(CreateChannelRequestByRankId(comm, topoInfo->userRank, peer, channelByRank, 1));
        channelDescs.insert(channelDescs.end(), channelByRank.begin(), channelByRank.end());
    }

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelBatchSendRecvMesh1D>(arg);
    };
    // 计算 self 在 actualPeers 中的索引
    uint32_t selfIdx = UINT32_MAX;
    for (uint32_t i = 0; i < actualPeers.size(); i++) {
        if (actualPeers[i] == static_cast<u32>(topoInfo->userRank)) {
            selfIdx = i;
            break;
        }
    }
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgBatchSendRecvMesh1D>(
        static_cast<uint64_t>(actualPeers.size()),
        selfIdx,
        param,
        subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempBatchSendRecvMesh1D][CalcRes] actualPeerCount[%zu], channelCount[%zu]",
        actualPeers.size(), channelDescs.size());
    return HCCL_SUCCESS;
}

void CcuTempBatchSendRecvMesh1D::SetBatchSendRecvInfo(const BatchSendRecvInfo &info)
{
    batchSendRecvInfo_ = info;
}

u64 CcuTempBatchSendRecvMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

HcclResult CcuTempBatchSendRecvMesh1D::ParseTasks()
{
    HCCL_INFO("[CcuTempBatchSendRecvMesh1D][ParseTasks] Start.");
    peerItems_.clear();
    actualPeers_.clear();
    maxRound_ = 0;

    // 按 peer 分组 send slices
    std::map<u32, std::vector<SendRecvSlice>> sendByPeer;
    for (auto &slice : batchSendRecvInfo_.sendSlices) {
        sendByPeer[slice.remoteRank].push_back(slice);
    }
    std::map<u32, std::vector<SendRecvSlice>> recvByPeer;
    for (auto &slice : batchSendRecvInfo_.recvSlices) {
        recvByPeer[slice.remoteRank].push_back(slice);
    }

    // 收集所有 peer（含 self）
    std::set<u32> peerSet;
    for (auto &kv : sendByPeer) {
        peerSet.insert(kv.first);
    }
    for (auto &kv : recvByPeer) {
        peerSet.insert(kv.first);
    }
    if (!batchSendRecvInfo_.sendToSelfSlices.empty() ||
        !batchSendRecvInfo_.recvFromSelfSlices.empty()) {
        peerSet.insert(static_cast<u32>(myRank_));
    }
    actualPeers_.assign(peerSet.begin(), peerSet.end());

    // 跨 rank peer：配对 send/recv，校验数量一致
    for (auto &kv : sendByPeer) {
        u32 peer = kv.first;
        auto &sends = kv.second;
        auto recvIt = recvByPeer.find(peer);
        if (recvIt == recvByPeer.end() || recvIt->second.size() != sends.size()) {
            HCCL_ERROR("[CcuTempBatchSendRecvMesh1D][ParseTasks] Send/recv item count mismatch "
                "for peer[%u], sendCount[%zu], recvCount[%zu]",
                peer, sends.size(), recvIt == recvByPeer.end() ? 0 : recvIt->second.size());
            return HCCL_E_PARA;
        }
        auto &recvs = recvIt->second;
        std::vector<PeerItemInfo> items;
        for (size_t i = 0; i < sends.size(); i++) {
            PeerItemInfo item;
            item.sendAddr = reinterpret_cast<uint64_t>(sends[i].addr);
            item.sendLen = sends[i].size;
            item.recvAddr = reinterpret_cast<uint64_t>(recvs[i].addr);
            items.push_back(item);
        }
        peerItems_[peer] = std::move(items);
    }

    // self peer：从 selfSlices 构建
    if (!batchSendRecvInfo_.sendToSelfSlices.empty()) {
        u32 selfRank = static_cast<u32>(myRank_);
        auto &selfSends = batchSendRecvInfo_.sendToSelfSlices;
        auto &selfRecvs = batchSendRecvInfo_.recvFromSelfSlices;
        std::vector<PeerItemInfo> items;
        for (size_t i = 0; i < selfSends.size(); i++) {
            PeerItemInfo item;
            item.sendAddr = reinterpret_cast<uint64_t>(selfSends[i].addr);
            item.sendLen = selfSends[i].size;
            item.recvAddr = reinterpret_cast<uint64_t>(selfRecvs[i].addr);
            items.push_back(item);
        }
        peerItems_[selfRank] = std::move(items);
    }

    // 计算 maxRound
    for (auto &kv : peerItems_) {
        if (kv.second.size() > maxRound_) {
            maxRound_ = kv.second.size();
        }
    }

    HCCL_INFO("[CcuTempBatchSendRecvMesh1D][ParseTasks] actualPeerCount[%zu], maxRound[%u]",
        actualPeers_.size(), maxRound_);
    return HCCL_SUCCESS;
}

HcclResult CcuTempBatchSendRecvMesh1D::KernelRun(const OpParam &param,
    const TemplateDataParams &templateDataParams,
    const TemplateResource &templateResource)
{
    HCCL_INFO("[CcuTempBatchSendRecvMesh1D][KernelRun] Start.");

    CHK_RET(ParseTasks());

    if (maxRound_ == 0) {
        HCCL_INFO("[CcuTempBatchSendRecvMesh1D][KernelRun] No tasks to process.");
        return HCCL_SUCCESS;
    }

    // 获取 token（使用第一个 send item 的地址所在内存区域）
    uint64_t token = 0;
    for (auto &kv : peerItems_) {
        if (!kv.second.empty()) {
            token = hcomm::CcuRep::GetTokenInfo(kv.second[0].sendAddr, 0);
            break;
        }
    }

    // 按 round 循环，每轮 launch 一次 kernel
    for (u32 round = 0; round < maxRound_; round++) {
        std::vector<PeerSendInfo> peerSendInfo;
        std::vector<uint64_t> peerRecvAddr;

        for (const u32 &peer : actualPeers_) {
            auto it = peerItems_.find(peer);
            if (it != peerItems_.end() && round < it->second.size()) {
                const PeerItemInfo &item = it->second[round];
                peerSendInfo.push_back({item.sendAddr, item.sendLen});
                peerRecvAddr.push_back(item.recvAddr);
            } else {
                peerSendInfo.push_back({0, 0});
                peerRecvAddr.push_back(0);
            }
        }

        std::unique_ptr<hcomm::CcuTaskArg> taskArg =
            std::make_unique<CcuTaskArgBatchSendRecvMesh1D>(token, peerSendInfo, peerRecvAddr);
        void *taskArgPtr = static_cast<void *>(taskArg.get());

        HCCL_DEBUG("[CcuTempBatchSendRecvMesh1D][KernelRun] round[%u/%u] launching kernel.",
            round, maxRound_);
        HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
            templateResource.ccuKernels[0], taskArgPtr);
    }

    HCCL_INFO("[CcuTempBatchSendRecvMesh1D][KernelRun] End.");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
