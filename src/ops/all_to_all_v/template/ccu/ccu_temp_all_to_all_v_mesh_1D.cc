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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh1d.h"
#include "ccu_temp_all_to_all_v_mesh_1D.h"

#define CONST_ZERO 0
#define CONST_ONE 1
#define CONST_TWO 2
#define CONST_THREE 3

constexpr u32 CCU_DIE_NUM_MAX_2 = 2;
constexpr u32 DIE_0 = 0;
constexpr u32 DIE_1 = 1;

namespace ops_hccl {

CcuTempAlltoAllVMesh1D::CcuTempAlltoAllVMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

CcuTempAlltoAllVMesh1D::~CcuTempAlltoAllVMesh1D()
{
}

HcclResult CcuTempAlltoAllVMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    uint32_t enableDieNum = 0;
    uint32_t enableDieId = 0;
    CHK_RET(GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc_, myRank_, enableDieNum, enableDieId));

    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) {
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D::CalcRes] invalid enableDieNum");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D::CalcRes] enableDieNum[%u], kernelNum[%u]", enableDieNum, kernelNum);

    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);

    for (const auto& pair : rankIdToChannelDesc_) {
        const std::vector<HcclChannelDesc>& channels = pair.second;
        for (const auto& channel : channels) {
            uint32_t channelDieId = 0;
            CHK_RET(GetChannelDieId(comm, myRank_, channel, channelDieId));
            if (enableDieNum == 1) { //单die
                if (channelDieId == enableDieId) {
                    channelsPerDie[DIE_0].push_back(channel);
                }
            } else { //双die
                if (channelDieId == DIE_0) {
                    channelsPerDie[DIE_0].push_back(channel);
                } else if (channelDieId == DIE_1) {
                    channelsPerDie[DIE_1].push_back(channel);
                }
            }
        }
    }

    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuKernelInfo kernelInfo;

        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                 return std::make_unique<CcuKernelAlltoAllVMesh1D>(arg);
                             };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgAlltoAllVMesh1D>(
            subCommRanks_[0].size(),
            mySubCommRank_,
            param.isMc2,
            kernelIdx,
            param,
            subCommRanks_
        );
        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

// executor在orchestra中调用
void CcuTempAlltoAllVMesh1D::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

// device侧调用
void CcuTempAlltoAllVMesh1D::InitInsAlgTemplate(
    std::vector<u64> &sendCounts, std::vector<u64> &recvCounts,
    std::vector<u64> &sdispls, std::vector<u64> &rdispls)
{
    sendCounts_ = sendCounts;
    recvCounts_ = recvCounts;
    sdispls_ = sdispls;
    rdispls_ = rdispls;

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, sendCounts[%u] is [%u]", i, sendCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, recvCounts[%u] is [%u]", i, recvCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, sdispls[%u] is [%u]", i, sdispls[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, rdispls[%u] is [%u]", i, rdispls[i]);
    }

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, sendCounts_[%u] is [%u]", i, sendCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, recvCounts_[%u] is [%u]", i, recvCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, sdispls_[%u] is [%u]", i, sdispls_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, rdispls_[%u] is [%u]", i, rdispls_[i]);
    }
}

HcclResult CcuTempAlltoAllVMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    HCCL_INFO("[CcuTempAlltoAllVMesh1D::FastLaunch] start");
    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    const uint64_t* args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;

    uint64_t rankSize_ = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs[5];
    HcclDataType dataType_ = param.all2AllVDataDes.sendType;
    uint64_t dataTypeSize_ =  SIZE_TABLE[dataType_];
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize_ * sizeof(u64),
    HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] param.varMemSize [%llu] is invalid", param.varMemSize), HCCL_E_PARA);
    
    A2ASendRecvInfo localSendRecvInfo;
    localSendRecvInfo.recvCounts.resize(rankSize_, 0);
    localSendRecvInfo.recvDispls.resize(rankSize_, 0);
    localSendRecvInfo.recvLength.resize(rankSize_, 0);
    localSendRecvInfo.recvOffset.resize(rankSize_, 0);
    localSendRecvInfo.sendCounts.resize(rankSize_, 0);
    localSendRecvInfo.sendDispls.resize(rankSize_, 0);
    localSendRecvInfo.sendLength.resize(rankSize_, 0);
    localSendRecvInfo.sendOffset.resize(rankSize_, 0);

    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_ ; i++) {
        u64 val = i / rankSize_;
 	    u64 curRank = i % rankSize_;
        switch(val) {
            case CONST_ZERO:
                localSendRecvInfo.sendLength[curRank] = data[i] * dataTypeSize_;
                break;
            case CONST_TWO:
                localSendRecvInfo.sendOffset[curRank] = data[i] * dataTypeSize_;
                break;
            case CONST_THREE:
                localSendRecvInfo.recvOffset[curRank] = data[i] * dataTypeSize_;
                break;
            default:
                break;
        }
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        const auto& cachedArgs = tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].cachedArgs;
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAlltoAllVMesh1D>(
            PointerToAddr(buffInfo_.inputPtr) + cachedArgs[0],
            PointerToAddr(buffInfo_.outputPtr) + cachedArgs[1],
            cachedArgs[2],
            cachedArgs[3],
            cachedArgs[4],
            cachedArgs[5],
            cachedArgs[6],
            localSendRecvInfo,
            cachedArgs[7],
            kernelIdx
        );

        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[kernelIdx],
            tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].kernelHandle, taskArgPtr));
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_INFO("[CcuTempAlltoAllVMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D::KernelRun(const OpParam& param, 
                                            const TemplateDataParams& templateDataParams,
                                            TemplateResource& templateResource)
{
    // 遗留：localSendRecvInfo_ 从哪里传入呢？
    HCCL_INFO("[CcuTempAlltoAllVMesh1D] KernelRun");

    buffInfo_ = templateDataParams.buffInfo;
    uint64_t totalSliceSize = localSendRecvInfo_.sendLength[0];

    if (tempRankSize_ == 1) {
        // ccu-alltoall算子的单P场景单独处理
        CHK_PRT_RET(localSendRecvInfo_.sendLength[myRank_] != localSendRecvInfo_.recvLength[myRank_],
                    HCCL_INFO("[CcuTempAlltoAllVMesh1D] rankSize = 1, sendLength[%llu] and recvLength[%llu]"
                               "should be equal.",
                               localSendRecvInfo_.sendLength[myRank_], localSendRecvInfo_.recvLength[myRank_]),
                    HcclResult::HCCL_E_PARA);
        CHK_PRT_RET(localSendRecvInfo_.sendLength[myRank_] == 0,
                    HCCL_INFO("[CcuTempAlltoAllVMesh1D] Single Rank and DataSlice size is 0, no need to process."),
                    HcclResult::HCCL_SUCCESS);

        DataSlice usrInSlice = DataSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, localSendRecvInfo_.sendLength[myRank_]);
        DataSlice usrOutSlice = DataSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, localSendRecvInfo_.sendLength[myRank_]);
        LocalCopy(templateResource.threads[0], usrInSlice, usrOutSlice);

        HCCL_INFO("[CcuTempAlltoAllVMesh1D] rankSize = 1, use InsLocalCopy for sliceSize[%llu].",
                  localSendRecvInfo_.sendLength[myRank_]);
        return HcclResult::HCCL_SUCCESS;
    }

    uint32_t                                rankId    = myRank_;

    std::vector<uint64_t> sliceSize;
    sliceSize.reserve(localSendRecvInfo_.sendLength.size());

    for (auto &slice : localSendRecvInfo_.sendLength) {
        sliceSize.push_back(slice);
    }

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t srcOffset = 0; // alltoallv假设src起始地址为发送rank的对应块起始地址
    uint64_t dstOffset = 0; // alltoallv假设dst起始地址为接收rank的对应块起始地址
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint32_t rankSize = tempRankSize_;

    HCCL_INFO("[CcuTempAllToAllVMesh1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        myRank_, tempRankSize_, inputAddr, outputAddr, sliceSize, srcOffset, dstOffset);

    uint32_t kernelNum = templateResource.ccuKernels.size();

    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    if (kernelNum == CCU_DIE_NUM_MAX_2) {
        CHK_RET(SplitDataFor2Dies(param, templateDataParams, die0Size, die1Size));
    } else {
        die0Size = templateDataParams.sliceSize;
    }

    HCCL_INFO("[CcuTempAlltoAllVMesh1D] die0Size=%llu, die1Size=%llu, kernelNum=%u",
              die0Size, die1Size, kernelNum);

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {
        if ((axisId == 0 && die0Size == 0) || (axisId == 1 && die1Size == 0)) {
            continue;
        }

        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAlltoAllVMesh1D>(
            inputAddr, outputAddr, token, srcOffset, dstOffset, rankSize, mySubCommRank_,
            localSendRecvInfo_, die0Size, axisId);

        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId],
            templateResource.ccuKernels[axisId], taskArgPtr));
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    CcuKernelSubmitInfo submitInfo;
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff,
        token, srcOffset, dstOffset, rankSize, mySubCommRank_, die0Size));//A2ASendRecvInfo
    
    for (u32 i = 0; i < kernelNum; i++) {
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }

    return HCCL_SUCCESS;
}

u64 CcuTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult CcuTempAlltoAllVMesh1D::SplitDataFor2Dies(const OpParam& param,
                                                         const TemplateDataParams& templateDataParams,
                                                         uint64_t& die0Size, uint64_t& die1Size) const
{
    uint64_t smallDataSize = 16 * 1024 * 1024;
    uint64_t typeSize = DataTypeSizeGet(param.all2AllVDataDes.sendType);
    uint64_t dataCount = templateDataParams.sliceSize / typeSize;

    if (templateDataParams.sliceSize < smallDataSize) {
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HCCL_SUCCESS;
    }

    u8 die0Port = 1;
    u8 die1Port = 1;
    die0Size = (dataCount * die0Port / (die0Port + die1Port)) * typeSize;
    die1Size = templateDataParams.sliceSize - die0Size;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl