/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh2die.h"
#include "ccu_temp_all_to_all_v_mesh2die.h"
#include "ccu_control_api.h"
#include "ccu_log.h"

namespace ops_hccl {

CcuTempAlltoAllVMesh2Die::CcuTempAlltoAllVMesh2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    channels_.resize(DIE_NUM);
    rankGroup_.resize(DIE_NUM);
}

CcuTempAlltoAllVMesh2Die::~CcuTempAlltoAllVMesh2Die()
{
}

HcclResult CcuTempAlltoAllVMesh2Die::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    CHK_PRT_RET(subCommRanks_.size() != 1 || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][CalcRes] Invalid subCommRanks[%u] or subCommRanks empty.",
            subCommRanks_.size()), HcclResult::HCCL_E_INTERNAL);

    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][CalcRes] rankSize[%u] subCommRanks0[%u].", templateRankSize_,
        subCommRanks_[0].size());

    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][CalcRes] channelDescs size[%u].", channelDescs.size());
    CHK_RET(PartitionChannels(comm, channelDescs));
    resourceRequest.channels.emplace_back(channelDescs);

    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuKernelInfo kernelInfo;
        strcpy(kernelInfo.kernelFuncName, "CcuAlltoAllVMesh2DieKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAlltoAllVMesh2DieKernel);

        const bool withMyRank = channels_[dieId].size() < channels_[1 - dieId].size();
        uint32_t localSize = rankGroup_[dieId].size();
        uint32_t localId = 0;
        for (uint32_t i = 0; i < rankGroup_[dieId].size(); i++) {
            if (rankGroup_[dieId][i] == myRank_) {
                localId = i;
                break;
            }
        }

        auto kernelArg = std::make_shared<CcuKernelArgAlltoAllVMesh2Die>();
        kernelArg->rankId = myRank_;
        kernelArg->withMyRank = withMyRank;
        kernelArg->localSize = localSize;
        kernelArg->localId = localId;
        kernelArg->peerSize = localSize;
        kernelArg->logicId = myRank_ % localSize;
        kernelArg->rankGroup.resize(localSize);
        for (uint32_t i = 0; i < localSize; i++) {
            kernelArg->rankGroup[i] = i;
        }
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][CalcRes] dieId=%u, channels=%llu, withMyRank=%u, localSize=%u, "
            "localId=%u, ccuKernelInfos=%llu", dieId, channels_[dieId].size(), withMyRank, localSize, localId,
            resourceRequest.ccuKernelInfos.size());
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2Die::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs)
{
    for (const auto &channel : channelDescs) {
        const RankId remoteRank = channel.remoteRank;
        uint32_t dieId = 0;
        HcclResult ret = GetChannelDieId(comm, myRank_, channel, dieId);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], Failed to "
                "get dieId. errNo[0x%016llx]", myRank_, remoteRank, HCCL_ERROR_CODE(ret)),
            ret);
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], dieId[%u] is "
                "invalid.", myRank_, remoteRank, dieId),
            HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], insert to "
            "channels at dieId[%u].", myRank_, remoteRank, dieId);
        channels_[dieId].emplace_back(channel);
        rankGroup_[dieId].push_back(remoteRank);
    }
    uint32_t minChannels = std::min(channels_[0].size(), channels_[1].size());
    uint32_t maxChannels = std::max(channels_[0].size(), channels_[1].size());
    CHK_PRT_RET(minChannels + 1 != maxChannels,
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d], Unexpected channels size, "
            "die0 channels[%u], die1 channels[%u].", myRank_, channels_[0].size(), channels_[1].size()),
        HcclResult::HCCL_E_PARA);
    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][PartitionChannels] Rank[%d], die0 channels[%u], die1 channels[%u].", myRank_,
        channels_[0].size(), channels_[1].size());
    if (channels_[0].size() < channels_[1].size()) {
        rankGroup_[0].push_back(myRank_);
    } else {
        rankGroup_[1].push_back(myRank_);
    }
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAlltoAllVMesh2Die::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

static std::vector<uint64_t> BuildTaskArgs(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
    const A2ASendRecvInfo &sendRecvInfo, const std::vector<uint32_t> &rankGroup,
    uint32_t localSize, uint32_t localId, const std::vector<uint32_t> &subCommRanks)
{
    std::vector<uint64_t> taskArgs;
    taskArgs.push_back(inputAddr);
    taskArgs.push_back(outputAddr);
    taskArgs.push_back(token);

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;

    uint64_t xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize, config);
    for (auto val : xnMaxTransportGoSize) {
        taskArgs.push_back(val);
    }

    std::map<uint32_t, uint32_t> userRankToSubCommRank;
    for (uint32_t i = 0; i < subCommRanks.size(); i++) {
        userRankToSubCommRank[subCommRanks[i]] = i;
    }

    for (uint32_t peerId = 0; peerId < localSize; peerId++) {
        uint32_t userRank = rankGroup[peerId];
        auto it = userRankToSubCommRank.find(userRank);
        uint32_t subCommRankId = (it != userRankToSubCommRank.end()) ? it->second : peerId;

        uint64_t sendLength = (subCommRankId < sendRecvInfo.sendLength.size()) ?
            sendRecvInfo.sendLength[subCommRankId] : 0;
        uint64_t sendOffset = (subCommRankId < sendRecvInfo.sendOffset.size()) ?
            sendRecvInfo.sendOffset[subCommRankId] : 0;
        uint64_t recvOffset = (subCommRankId < sendRecvInfo.recvOffset.size()) ?
            sendRecvInfo.recvOffset[subCommRankId] : 0;

        uint64_t sendTailSize = sendLength % UB_MAX_TRANS_SIZE;
        uint64_t sendLoopNum = UINT64_MAX - 1 - (sendLength / UB_MAX_TRANS_SIZE);

        taskArgs.push_back(sendOffset);
        taskArgs.push_back(recvOffset);
        taskArgs.push_back(sendTailSize);

        auto tailGoSize = CalGoSize(sendTailSize, config);
        for (auto val : tailGoSize) {
            taskArgs.push_back(val);
        }

        taskArgs.push_back(sendLoopNum);
    }

    return taskArgs;
}

HcclResult CcuTempAlltoAllVMesh2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource& templateResource)
{
    CHK_PRT_RET(subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);

    const auto &buffInfo = templateDataParams.buffInfo;
    CHK_PRT_RET(buffInfo.inputPtr == nullptr || buffInfo.outputPtr == nullptr,
        HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] Rank[%d] input[%#llx] or output[%#llx] is null",
            myRank_, buffInfo.inputPtr, buffInfo.outputPtr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = PointerToAddr(buffInfo.inputPtr) + buffInfo.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo.outputPtr) + buffInfo.outBuffBaseOff;
    HCCL_INFO("[CcuTempAlltoAllVMesh2Die][KernelRun] begin. Rank[%d], input[%#llx], output[%#llx], "
        "sendType[%d], recvType[%d]", myRank_, inputAddr, outputAddr,
        param.all2AllVDataDes.sendType, param.all2AllVDataDes.recvType);

    uint64_t token;
    CHK_RET(GetToken(buffInfo, token));

    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        uint32_t localSize = rankGroup_[dieId].size();
        uint32_t localId = 0;
        for (uint32_t i = 0; i < rankGroup_[dieId].size(); i++) {
            if (rankGroup_[dieId][i] == myRank_) {
                localId = i;
                break;
            }
        }

        std::vector<uint64_t> taskArgs = BuildTaskArgs(inputAddr, outputAddr, token,
            localSendRecvInfo_, rankGroup_[dieId], localSize, localId, subCommRanks_[0]);
        uint64_t argSize = taskArgs.size();

        HCCL_INFO("[CcuTempAlltoAllVMesh2Die][KernelRun] dieId=%u, localSize=%u, localId=%u, argSize=%llu",
            dieId, localSize, localId, argSize);

        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieId],
            templateResource.ccuKernels[dieId], taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuKernelSubmitInfo subCommInfo;
        subCommInfo.kernelHandle = templateResource.ccuKernels[dieId];
        CHK_RET(FillCachedArgs(subCommInfo, inputAddr, outputAddr, token,
            buffInfo.inBuffBaseOff, buffInfo.outBuffBaseOff));
        templateResource.submitInfos.push_back(subCommInfo);
    }

    HCCL_DEBUG("[CcuTempAlltoAllVMesh2Die][KernelRun] end. Rank[%d]", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2Die::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAlltoAllVMesh2Die::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[CcuTempAlltoAllVMesh2Die::FastLaunch] start");

    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    buffInfo_ = tempFastLaunchCtx.buffInfo;

    std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
    if (kernelNum > 1) {
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        uint32_t localSize = rankGroup_[dieId].size();
        uint32_t localId = 0;
        for (uint32_t i = 0; i < rankGroup_[dieId].size(); i++) {
            if (rankGroup_[dieId][i] == myRank_) {
                localId = i;
                break;
            }
        }

        std::vector<uint64_t> taskArgs = BuildTaskArgs(inputAddr, outputAddr, token,
            localSendRecvInfo_, rankGroup_[dieId], localSize, localId, subCommRanks_[0]);
        uint64_t argSize = taskArgs.size();

        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[dieId],
            tempFastLaunchCtx.ccuKernelSubmitInfos[dieId].kernelHandle,
            taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllVMesh2Die::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    if (kernelNum > 1) {
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_INFO("[CcuTempAlltoAllVMesh2Die::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAlltoAllVMesh2Die::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

}
