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
#include "kernel/ccu_kernel_all_to_all_mesh2die.h"
#include "ccu_temp_all_to_all_mesh2die.h"
#include "ccu_control_api.h"
#include "ccu_log.h"

namespace ops_hccl {

CcuTempAllToAllMesh2Die::CcuTempAllToAllMesh2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    channels_.resize(DIE_NUM);
    rankGroup_.resize(DIE_NUM);
}

CcuTempAllToAllMesh2Die::~CcuTempAllToAllMesh2Die()
{
}

HcclResult CcuTempAllToAllMesh2Die::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(PartitionChannels(comm, channelDescs));
    resourceRequest.channels.emplace_back(channelDescs);

    const uint32_t rankSize = subCommRanks_[0].size();
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuKernelInfo kernelInfo;
        strcpy(kernelInfo.kernelFuncName, "CcuAllToAllMesh2DieKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllToAllMesh2DieKernel);

        const bool withMyRank = channels_[dieId].size() < channels_[1 - dieId].size();
        uint32_t localSize = rankGroup_[dieId].size();
        uint32_t localId = 0;
        for (uint32_t i = 0; i < rankGroup_[dieId].size(); i++) {
            if (rankGroup_[dieId][i] == myRank_) {
                localId = i;
                break;
            }
        }

        auto kernelArg = std::make_shared<CcuKernelArgAllToAllMesh2Die>();
        kernelArg->rankSize = rankSize;
        kernelArg->rankId = myRank_;
        kernelArg->withMyRank = withMyRank;
        kernelArg->localSize = localSize;
        kernelArg->localId = localId;
        kernelArg->rankGroup = rankGroup_[dieId];
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAllToAllMesh2Die][CalcRes] dieId=%u, channels=%llu, withMyRank=%u, localSize=%u, "
            "localId=%u, ccuKernelInfos=%llu", dieId, channels_[dieId].size(), withMyRank, localSize, localId,
            resourceRequest.ccuKernelInfos.size());
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh2Die::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs)
{
    for (const auto &channel : channelDescs) {
        const RankId remoteRank = channel.remoteRank;
        uint32_t dieId = 0;
        HcclResult ret = GetChannelDieId(comm, myRank_, channel, dieId);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CcuTempAllToAllMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], Failed to "
                "get dieId. errNo[0x%016llx]", myRank_, remoteRank, HCCL_ERROR_CODE(ret)),
            ret);
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempAllToAllMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], dieId[%u] is "
                "invalid.", myRank_, remoteRank, dieId),
            HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempAllToAllMesh2Die][PartitionChannels] Rank[%d] channel to remoteRank[%d], insert to "
            "channels at dieId[%u].", myRank_, remoteRank, dieId);
        channels_[dieId].emplace_back(channel);
        rankGroup_[dieId].push_back(remoteRank);
    }
    uint32_t minChannels = std::min(channels_[0].size(), channels_[1].size());
    uint32_t maxChannels = std::max(channels_[0].size(), channels_[1].size());
    CHK_PRT_RET(minChannels + 1 != maxChannels,
        HCCL_ERROR("[CcuTempAllToAllMesh2Die][PartitionChannels] Rank[%d], Unexpected channels size, "
            "die0 channels[%u], die1 channels[%u].", myRank_, channels_[0].size(), channels_[1].size()),
        HcclResult::HCCL_E_PARA);
    HCCL_DEBUG("[CcuTempAllToAllMesh2Die][PartitionChannels] Rank[%d], die0 channels[%u], die1 channels[%u].", myRank_,
        channels_[0].size(), channels_[1].size());
    rankGroup_[0].push_back(myRank_);
    rankGroup_[1].push_back(myRank_);
    return HcclResult::HCCL_SUCCESS;
}

static std::vector<uint64_t> BuildTaskArgs(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
    uint64_t sliceSize, uint64_t inputSliceStride, uint64_t outputoffset,
    const std::vector<uint32_t> &rankGroup, uint32_t localSize)
{
    std::vector<uint64_t> taskArgs;
    taskArgs.push_back(inputAddr);
    taskArgs.push_back(outputAddr);
    taskArgs.push_back(token);
    taskArgs.push_back(sliceSize);
    taskArgs.push_back(inputSliceStride);
    taskArgs.push_back(outputoffset);

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;

    auto goSize = CalGoSize(sliceSize, config);
    for (auto val : goSize) {
        taskArgs.push_back(val);
    }

    for (uint32_t i = 0; i < localSize; i++) {
        uint64_t inputOffset = inputSliceStride * rankGroup[i];
        taskArgs.push_back(inputOffset);
    }

    return taskArgs;
}

HcclResult CcuTempAllToAllMesh2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllToAllMesh2Die][KernelRun] begin");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t sliceSize = templateDataParams.sliceSize;
    uint64_t inputSliceStride = templateDataParams.sdispls[1] *
        DATATYPE_SIZE_TABLE[param.all2AllDataDes.recvType] - buffInfo_.inBuffBaseOff;
    uint64_t outputSliceStride = templateDataParams.sdispls[1] *
        DATATYPE_SIZE_TABLE[param.all2AllDataDes.recvType] - buffInfo_.inBuffBaseOff;
    uint64_t outputoffset = outputSliceStride * myRank_;

    HCCL_INFO("[CcuTempAllToAllMesh2Die][KernelRun] Rank[%d], input[%#llx], output[%#llx], "
        "sliceSize[%llu], inputSliceStride[%llu], outputoffset[%llu]",
        myRank_, inputAddr, outputAddr, sliceSize, inputSliceStride, outputoffset);

    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        uint32_t localSize = rankGroup_[dieId].size();

        std::vector<uint64_t> taskArgs = BuildTaskArgs(inputAddr, outputAddr, token,
            sliceSize, inputSliceStride, outputoffset, rankGroup_[dieId], localSize);
        uint64_t argSize = taskArgs.size();

        HCCL_INFO("[CcuTempAllToAllMesh2Die][KernelRun] dieId=%u, localSize=%u, argSize=%llu",
            dieId, localSize, argSize);

        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieId],
            templateResource.ccuKernels[dieId], taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllToAllMesh2Die][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuKernelSubmitInfo subCommInfo;
        subCommInfo.kernelHandle = templateResource.ccuKernels[dieId];
        CHK_RET(FillCachedArgs(subCommInfo, inputAddr, outputAddr, token,
            buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
        templateResource.submitInfos.push_back(subCommInfo);
    }

    HCCL_DEBUG("[CcuTempAllToAllMesh2Die][KernelRun] end. Rank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh2Die::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllToAllMesh2Die::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[CcuTempAllToAllMesh2Die::FastLaunch] start");

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

    const uint64_t *cachedArgs = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    uint64_t sliceSize = cachedArgs[3];
    uint64_t inputSliceStride = cachedArgs[4];
    uint64_t outputoffset = cachedArgs[5];

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        uint32_t localSize = rankGroup_[dieId].size();

        std::vector<uint64_t> taskArgs = BuildTaskArgs(inputAddr, outputAddr, token,
            sliceSize, inputSliceStride, outputoffset, rankGroup_[dieId], localSize);
        uint64_t argSize = taskArgs.size();

        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[dieId],
            tempFastLaunchCtx.ccuKernelSubmitInfos[dieId].kernelHandle,
            taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllToAllMesh2Die::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    if (kernelNum > 1) {
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_INFO("[CcuTempAllToAllMesh2Die::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllToAllMesh2Die::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

}
