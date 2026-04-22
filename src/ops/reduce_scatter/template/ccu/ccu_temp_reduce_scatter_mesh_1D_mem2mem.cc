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
#include "ccu_kernel_reduce_scatter_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"

namespace ops_hccl {

CcuTempReduceScatterMesh1DMem2Mem::CcuTempReduceScatterMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                     const std::vector<std::vector<u32>>& subCommRanks)
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

CcuTempReduceScatterMesh1DMem2Mem::~CcuTempReduceScatterMesh1DMem2Mem()
{
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");
    }
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    // 1.从获得的channelDesc，判断kernel发送到几个die上
    uint32_t enableDieNum = 0;
    uint32_t enableDieId = 0;
    CHK_RET(GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc_, myRank_, enableDieNum, enableDieId));
    
    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) { // 目前只支持1个或2个die
        HCCL_ERROR("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] get channelDescs fail");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    // 无论几个kernel，都创建2条流
    resourceRequest.sfsfdsfdsfdsafdf = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 2.将channelDescs分到2个die
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    
    constexpr u32 DIE_NUM_1 = 1;
    constexpr u32 DIE_NUM_2 = 2;
    constexpr u32 DIE_0 = 0;
    constexpr u32 DIE_1 = 1;
    for (const auto &pair : rankIdToChannelDesc_) {
        u32 remoteRankId = pair.first;
        if (enableDieNum == DIE_NUM_1) {
            // 单die情况，所有channel都分配给die0
            channelsPerDie[DIE_0] = channelDescs;
            CHK_RET(SelectChannelToVec(comm, myRank_, remoteRankId, rankIdToChannelDesc_, enableDieId, rank2ChannelIdx,
                channelsPerDie[DIE_0]));
        } else if (enableDieNum == DIE_NUM_2) {
            CHK_RET(SelectChannelToVec(
                comm, myRank_, remoteRankId, rankIdToChannelDesc_, DIE_0, rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(SelectChannelToVec(
                comm, myRank_, remoteRankId, rankIdToChannelDesc_, DIE_1, rank2ChannelIdx, channelsPerDie[DIE_1]));
        }
    }

    // 3.构造kernelInfo
    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                 return std::make_unique<CcuKernelReduceScatterMesh1DMem2Mem>(arg);
                             };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1DMem2Mem>(subCommRanks_[0].size(),
                                                                                        mySubCommRank_,
                                                                                        param, 
                                                                                        kernelIdx,
                                                                                        subCommRanks_);
        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::FastLaunch] start");
    const uint64_t *args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    buffInfo_ = tempFastLaunchCtx.buffInfo;

    // 前流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuTaskArgReduceScatterMesh1DMem2Mem taskArg(
        PointerToAddr(buffInfo_.inputPtr) + args[0],
        PointerToAddr(buffInfo_.outputPtr) + args[1],
        args[2], 
        PointerToAddr(buffInfo_.hcclBuff.addr) + args[3], 
        args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);

        void* taskArgPtr = static_cast<void*>(&taskArg);

        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0], 
        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));
    }
    
    // 后流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::SplitDataFor2Dies(const OpParam& param, const uint64_t sliceSize,
                                                               uint64_t& die0Size, uint64_t& die1Size) const
{
    constexpr uint64_t sizeBoundary = 16 * 1024;
    uint64_t typeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t dataCount = sliceSize / typeSize;

    if (sliceSize <= sizeBoundary) {   // 数据量极小，不划分die
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HcclResult::HCCL_SUCCESS;
    }
    u8 die0BWcoeff = 1;
    u8 die1BWcoeff = 1;

    die0Size = (dataCount * die0BWcoeff / (die0BWcoeff + die1BWcoeff)) * typeSize;
    die1Size = sliceSize - die0Size;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[CcuTempReduceScatterMesh1DMem2Mem] sliceSize is 0, no need to do, just success.");
        return HCCL_SUCCESS;
    }
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t repeatNumTmp       = templateDataParams.repeatNum;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;

    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;

    u32 kernelNum = templateResource.ccuKernels.size();
    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    uint64_t die0LastSliceSize = 0;
    uint64_t die1LastSliceSize = 0;
    constexpr uint32_t MAX_DIE_NUM_2 = 2;
    if (kernelNum == MAX_DIE_NUM_2) {
        SplitDataFor2Dies(param, templateDataParams.sliceSize, die0Size, die1Size);
        SplitDataFor2Dies(param, templateDataParams.tailSize, die0LastSliceSize, die1LastSliceSize);
    } else {
        die0Size = templateDataParams.sliceSize;
        die0LastSliceSize = templateDataParams.tailSize;
    }

    // 前流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {
        if ((axisId == 0 && die0Size == 0 && die0LastSliceSize == 0)
            || (axisId == 1 && die1Size == 0 && die1LastSliceSize == 0)) {
            // 数据长度为0的kernel不下发
            continue;
        }
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceScatterMesh1DMem2Mem>(
            inputAddr, outputAddr, token, scratchAddr, inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride,
            die0Size, die1Size, die0LastSliceSize, die1LastSliceSize, repeatNum);

        void* taskArgPtr = static_cast<void*>(taskArg.get());

        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId], templateResource.ccuKernels[axisId], taskArgPtr));
    }
    
    // 后流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token, buffInfo_.hcclBuffBaseOff,
        inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride, die0Size, 
        die1Size, die0LastSliceSize, die1LastSliceSize, repeatNum));
    for (u32 i = 0; i < kernelNum; i++) { 
        // 2个kernel的TaskArg相同
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }

    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 CcuTempReduceScatterMesh1DMem2Mem::GetThreadNum() const
{
    constexpr uint32_t KERNEL_NUM_2 = 2;
    return KERNEL_NUM_2;
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;

    return HCCL_SUCCESS;
}
} // namespace ops_hccl