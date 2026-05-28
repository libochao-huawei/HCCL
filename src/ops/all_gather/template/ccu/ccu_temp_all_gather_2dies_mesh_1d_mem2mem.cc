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
#include "ccu_kernel_all_gather_2dies_mesh1d_mem2mem.h"
#include "ccu_temp_all_gather_2dies_mesh_1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch.h"

namespace ops_hccl {
#define ALL_GATHER_2DIES_M2M_THREAD_NUM 2

CcuTempAllGather2DiesMeshMem2Mem1D::CcuTempAllGather2DiesMeshMem2Mem1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}
 
CcuTempAllGather2DiesMeshMem2Mem1D::~CcuTempAllGather2DiesMeshMem2Mem1D()
{
}
 
HcclResult CcuTempAllGather2DiesMeshMem2Mem1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                       AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(ALL_GATHER_2DIES_M2M_THREAD_NUM);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    uint32_t rankId = mySubCommRank_;
    EndpointAttrDieId tmpDieId {};
    HCCL_DEBUG("[CcuTempAllGather2DiesMeshMem2Mem1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    std::vector<uint32_t> rankIdGroup0, rankIdGroup1;
    bool if0HandleSelfRank = true;
    CcuKernelInfo kernelInfo0, kernelInfo1;
    strcpy(kernelInfo0.kernelFuncName, "CcuAllGather2DiesMesh1DMem2MemKernel");
    kernelInfo0.kernelFunc = reinterpret_cast<void *>(CcuAllGather2DiesMesh1DMem2MemKernel);
    strcpy(kernelInfo1.kernelFuncName, "CcuAllGather2DiesMesh1DMem2MemKernel");
    kernelInfo1.kernelFunc = reinterpret_cast<void *>(CcuAllGather2DiesMesh1DMem2MemKernel);

    for (u32 j = 0; j < channelDescs.size(); j++) {
        CHK_RET(GetChannelDieId(comm, rankId, channelDescs[j], tmpDieId));
        if (tmpDieId == 0) {
            kernelInfo0.channels.push_back(channelDescs[j]);
            rankIdGroup0.push_back(channelDescs[j].remoteRank);
        } else {
            kernelInfo1.channels.push_back(channelDescs[j]);
            rankIdGroup1.push_back(channelDescs[j].remoteRank);
        }
    }
    if ((rankIdGroup0.size() > rankIdGroup1.size() && rankIdGroup1.size() != 0) || rankIdGroup0.size() == 0) {
        if0HandleSelfRank = false;
    }
    if (rankIdGroup0.size() != 0) {
        auto kernelArg0 = std::make_shared<CcuKernelArgAllGather2DiesMesh1DMem2Mem>();
        kernelArg0->dimSize = subCommRanks_[0].size();
        kernelArg0->rankId = rankId;
        kernelArg0->rankIdGroup = rankIdGroup0;
        kernelArg0->ifHandleSelfRank = if0HandleSelfRank;
        kernelArg0->opParam = param;
        kernelArg0->subCommRanks = subCommRanks_;
        kernelInfo0.setKernelArg(kernelArg0);
    }
    if (rankIdGroup1.size() != 0) {
        auto kernelArg1 = std::make_shared<CcuKernelArgAllGather2DiesMesh1DMem2Mem>();
        kernelArg1->dimSize = subCommRanks_[0].size();
        kernelArg1->rankId = rankId;
        kernelArg1->rankIdGroup = rankIdGroup1;
        kernelArg1->ifHandleSelfRank = !if0HandleSelfRank;
        kernelArg1->opParam = param;
        kernelArg1->subCommRanks = subCommRanks_;
        kernelInfo1.setKernelArg(kernelArg1);
    }
    if (rankIdGroup0.size() != 0) {
        resourceRequest.ccuKernelInfos.push_back(kernelInfo0);
    }
    if (rankIdGroup1.size() != 0) {
        resourceRequest.ccuKernelInfos.push_back(kernelInfo1);  
    }
    HCCL_DEBUG("[CcuTempAllGather2DiesMeshMem2Mem1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult CcuTempAllGather2DiesMeshMem2Mem1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllGather2DiesMeshMem2Mem1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllGather2DiesMeshMem2Mem1D::FastLaunch] start");

    for (uint64_t i = 0; i < ALL_GATHER_2DIES_M2M_THREAD_NUM; i++) {
        uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[i].cachedArgs);
        constexpr u32 inputIdx = 0;
        constexpr u32 outputIdx = 1;
        constexpr u32 sliceSizeIdx = 2;
        constexpr u32 offSetIdx = 3;
        constexpr u32 isInputOutputEqualIdx = 5;
        constexpr u32 inputOffsetIdx = 10;
        constexpr u32 outputOffsetIdx = 11;
        uint64_t argSize = 10;

        uint64_t inputAddr  = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
        uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
        uint64_t sliceSize  = args[sliceSizeIdx];
        uint64_t offSet     = args[offSetIdx];
        uint64_t currentRankSliceInputOffset = inputAddr + mySubCommRank_ * sliceSize;
        uint64_t currentRankSliceOutputOffset = outputAddr + offSet;
        bool inputOutputEqual = (currentRankSliceInputOffset == currentRankSliceOutputOffset);
        uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);

        args[inputIdx]  = inputAddr;
        args[outputIdx] = outputAddr;
        args[isInputOutputEqualIdx] = isInputOutputEqual;

        void *taskArgs = reinterpret_cast<void*>(args);
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[i],
                                                   tempFastLaunchCtx.ccuKernelSubmitInfos[i].kernelHandle,
                                                   taskArgs, argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGather2DiesMeshMem2Mem1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    HCCL_DEBUG("[CcuTempAllGather2DiesMeshMem2Mem1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGather2DiesMeshMem2Mem1D::KernelRun(const OpParam& param,
                                                         const TemplateDataParams& templateDataParams,
                                                         TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    uint32_t rankId = mySubCommRank_;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t sliceSize = templateDataParams.sliceSize;
    uint64_t offSet = rankId * templateDataParams.outputSliceStride;

    bool inputOutputEqual = (inputAddr + rankId * templateDataParams.inputSliceStride == outputAddr + offSet);
    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(sliceSize, config);

    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint64_t i = 0; i < ALL_GATHER_2DIES_M2M_THREAD_NUM; i++) {
        std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, sliceSize, offSet, token,
                                          isInputOutputEqual, goSize[0], goSize[1], goSize[2], goSize[3]};
        uint64_t argSize = 10;

        HCCL_INFO("[CcuTempAllGather2DiesMeshMem2Mem1D::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
                   "sliceSize[%llu], offSet[%llu], isInputOutputEqual[%llu]",
                   inputAddr, outputAddr, sliceSize, offSet, isInputOutputEqual);

        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[i], templateResource.ccuKernels[i],
                                                    taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGather2DiesMeshMem2Mem1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, sliceSize, offSet, token,
                               isInputOutputEqual, goSize[0], goSize[1], goSize[2], goSize[3],
                               buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
        templateResource.submitInfos.push_back(submitInfo);

        HCCL_DEBUG("[CcuTempAllGather2DiesMeshMem2Mem1D::KernelRun] end");
    }

    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    return HcclResult::HCCL_SUCCESS;
}
 
u64 CcuTempAllGather2DiesMeshMem2Mem1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}
 
u64 CcuTempAllGather2DiesMeshMem2Mem1D::GetThreadNum() const
{
    return ALL_GATHER_2DIES_M2M_THREAD_NUM;
}

HcclResult CcuTempAllGather2DiesMeshMem2Mem1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}
 
}// namespace ops_hccl