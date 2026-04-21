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
#include "ccu_kernel_all_gather_mesh1d.h"
#include "ccu_temp_all_gather_mesh_1D.h"

namespace ops_hccl {

CcuTempAllGatherMesh1D::CcuTempAllGatherMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllGatherMesh1D::~CcuTempAllGatherMesh1D()
{
}

HcclResult CcuTempAllGatherMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;

    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : channelDescs){
            if(channel.channelProtocol != COMM_PROTOCOL_UBC_CTP){
                HCCL_ERROR("[CcuTempAllGatherMesh1D][CalcRes] channelProtocol: %u", channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    uint32_t enableDieNum = 0;
    uint32_t enableDieId = 0;
    CHK_RET(GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc_, myRank_, enableDieNum, enableDieId));

    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) {
        HCCL_ERROR("[CcuTempAllGatherMesh1D::CalcRes] invalid enableDieNum");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    HCCL_DEBUG("[CcuTempAllGatherMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u] enableDieNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, enableDieNum);

    constexpr u32 DIE_0 = 0;
    constexpr u32 DIE_1 = 1;
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);

    // For Mesh1D: collect channels per die
    for (const auto& pair : rankIdToChannelDesc_) {
        const std::vector<HcclChannelDesc>& channels = pair.second;
        for (const auto& channel : channels) {
            uint32_t channelDieId = 0;
            CHK_RET(GetChannelDieId(comm, myRank_, channel, channelDieId));
            if (enableDieNum == 1) {
                // Single die mode: select channels from enableDieId
                if (channelDieId == enableDieId) {
                    channelsPerDie[DIE_0].push_back(channel);
                }
            } else {
                // Double die mode: select channels from each die
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
                                 return std::make_unique<CcuKernelAllGatherMesh1D>(arg);
                             };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1D>(subCommRanks_[0].size(),
                                                                                     mySubCommRank_,
                                                                                     kernelIdx,
                                                                                     param,
                                                                                     subCommRanks_);

        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[CcuTempAllGatherMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    HCCL_DEBUG("[CcuTempAllGatherMesh1D::FastLaunch] start");
    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    const uint64_t* args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuTaskArgAllGatherMesh1D taskArg(
            PointerToAddr(buffInfo_.inputPtr) + args[0],
            PointerToAddr(buffInfo_.outputPtr) + args[1],
            args[2], args[3], args[4], args[5]);

        void* taskArgPointer = static_cast<void*>(&taskArg);
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[kernelIdx],
            tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].kernelHandle, taskArgPointer));
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_DEBUG("[CcuTempAllGatherMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1D::KernelRun(const OpParam& param,
                                             const TemplateDataParams& templateDataParams,
                                             TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint32_t rankId    = mySubCommRank_;
    uint64_t offset    = rankId * templateDataParams.outputSliceStride;;

    uint64_t sliceSize = templateDataParams.sliceSize;  // 获取本rank需要处理的数据量

    HcclDataType dataType       = param.DataDes.dataType;
    uint64_t dataTypeSize       = DataTypeSizeGet(dataType);
    uint64_t dataCount          = sliceSize / dataTypeSize;
    if (dataCount == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1D] DataCount == 0, Template Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    uint32_t kernelNum = templateResource.ccuKernels.size();

    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    constexpr uint32_t MAX_DIE_NUM_2 = 2;
    if (kernelNum == MAX_DIE_NUM_2) {
        CHK_RET(SplitDataFor2Dies(param, templateDataParams, die0Size, die1Size));
    } else {
        die0Size = templateDataParams.sliceSize;
    }

    HCCL_INFO("[CcuTempAllGatherMesh1D::KernelRun] die0Size [%llu], die1Size [%llu], kernelNum[%u]",
              die0Size, die1Size, kernelNum);

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {
        if ((templateDataParams.tailSize == 0) && ((axisId == 0 && die0Size == 0) || (axisId == 1 && die1Size == 0))) {
            continue;
        }
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGatherMesh1D>(
            inputAddr, outputAddr, token, offset, die0Size, die1Size);

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
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token, offset,
        die0Size, die1Size));
    for (u32 i = 0; i < kernelNum; i++) {
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }

    HCCL_INFO("[CcuTempAllGatherMesh1D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherMesh1D::GetThreadNum() const
{
    return 1;
}
 
HcclResult CcuTempAllGatherMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1D::SplitDataFor2Dies(const OpParam& param,
                                                         const TemplateDataParams& templateDataParams,
                                                         uint64_t& die0Size, uint64_t& die1Size) const
{
    uint64_t smallDataSize = 16 * 1024;

    constexpr uint64_t MULTIPLIER = 4;
    uint64_t typeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t dataCount = (templateDataParams.sliceSize / typeSize);

    if (templateDataParams.sliceSize < smallDataSize) {
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HcclResult::HCCL_SUCCESS;
    }

    u8 die0PortGroupSize = 1;
    u8 die1PortGroupSize = 1;

    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * typeSize;
    die1Size = templateDataParams.sliceSize - die0Size;
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl