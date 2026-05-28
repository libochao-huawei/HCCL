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
#include "ccu_kernel_reduce_scatter_mesh1d_2die_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_2die_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch.h"
 
namespace ops_hccl {
 
constexpr u32 DIE_NUM = 2;
constexpr u32 UDIE0 = 0;
constexpr u32 UDIE1 = 1;
 
CcuTempReduceScatterMeshMem2Mem1D2Die::CcuTempReduceScatterMeshMem2Mem1D2Die(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}
 
CcuTempReduceScatterMeshMem2Mem1D2Die::~CcuTempReduceScatterMeshMem2Mem1D2Die()
{
}
 
HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u] ccuKernelNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, resourceRequest.ccuKernelNum[0]);
    
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<std::vector<HcclChannelDesc>> channelDescsVec;
    std::vector<std::vector<u32>>             subRankGroup;
    bool isReduceToOutput = false;
 
    channelDescsVec.resize(DIE_NUM);
    subRankGroup.resize(DIE_NUM);
 
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
 
    for (auto channel : channelDescs) {
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempReduceScatterMeshMem2Mem1D2Die][CalcRes] dieId is invalid"), HCCL_E_INTERNAL);
        channelDescsVec[dieId].push_back(channel);
        subRankGroup[dieId].push_back(channel.remoteRank);
    }
 
    subRankGroup[UDIE1].push_back(myRank_);
 
    uint32_t tmpDieId = myRank_ < subRankGroup[UDIE0].back() ? 0 : 1;
    localReduceOffset_ = subRankGroup[1 - tmpDieId][0];
 
    for (int dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuKernelInfo kernelInfo;
        strcpy(kernelInfo.kernelFuncName, "CcuReduceScatterMesh1D2DieMem2MemKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceScatterMesh1D2DieMem2MemKernel);
 
        isReduceToOutput = channelDescsVec[dieId].size() > channelDescsVec[1 - dieId].size() ? false : true;
        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1D2DieMem2Mem>();
        kernelArg->gRankSize = templateRankSize_;
        kernelArg->rankSize = subRankGroup[dieId].size();
        kernelArg->isReduceToOutput = isReduceToOutput;
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subRankGroup = subRankGroup[dieId];
        kernelArg->subCommRanks = subCommRanks_;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channelDescsVec[dieId];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }
 
    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
 
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceScatterMeshMem2Mem1D2Die::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D2Die::FastLaunch] start");

    for (uint64_t dieIdx = 0; dieIdx < DIE_NUM; dieIdx++) {
        uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[dieIdx].cachedArgs);
        constexpr u32 inputIdx = 0;
        constexpr u32 outputIdx = 1;
        constexpr u32 scratchIdx = 3;
        constexpr u32 inputOffsetIdx = 4;
        constexpr u32 outputOffsetIdx = 5;
        constexpr u32 scratchOffsetIdx = 18;
        uint64_t argSize = 15;

        args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
        args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
        args[scratchIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr) + args[scratchOffsetIdx];

        void *taskArgs = reinterpret_cast<void*>(args);
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[dieIdx],
                                                   tempFastLaunchCtx.ccuKernelSubmitInfos[dieIdx].kernelHandle,
                                                   taskArgs, argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempReduceScatterMeshMem2Mem1D2Die::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D2Die::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[CcuTempReduceScatterMeshMem2Mem1D2Die] sliceSize is 0, no need to do, just success.");
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
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize    = templateDataParams.sliceSize;
    uint64_t lastSliceSize      = templateDataParams.tailSize;
 
    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;
    uint64_t currentRankSliceInputOffset = inputSliceStride * mySubCommRank_;
    uint64_t currentRankSliceOutputOffset = templateDataParams.outputSliceStride * mySubCommRank_;
 
    LoopGroupConfig config{};
    config.msInterleave = REDUCE_2DIE_MS_CNT;
    config.loopCount    = 16;
    config.memSlice     = CCU_MS_SIZE;
    auto goSize = CalGoSize(normalSliceSize, config);
    
    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for(uint64_t dieIdx = 0; dieIdx < DIE_NUM; dieIdx++) {
        std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, scratchAddr,
                                          currentRankSliceInputOffset, currentRankSliceOutputOffset,
                                          inputRepeatStride, outputRepeatStride,
                                          normalSliceSize, lastSliceSize, repeatNum,
                                          goSize[0], goSize[1], goSize[2], goSize[3]};

        HCCL_INFO("[CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
                  "scratchAddr[%llu], inputOffset[%llu], outputOffset[%llu], "
                  "inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], repeatNum[%llu]",
                  inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset,
                  inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumTmp);

        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieIdx], templateResource.ccuKernels[dieIdx],
                                                    taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[dieIdx];
        CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, token, scratchAddr,
            currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride,
            normalSliceSize, lastSliceSize, repeatNum, goSize[0], goSize[1], goSize[2], goSize[3],
            buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, buffInfo_.hcclBuffBaseOff));
        templateResource.submitInfos.push_back(submitInfo);
    }
    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
 
    // 使用SDMA inlineReduce做local copy scratch -> output
    uint64_t hcclBuffOffset = buffInfo_.hcclBuffBaseOff;
    hcclBuffOffset += normalSliceSize * (templateRankSize_ / DIE_NUM);
    DataSlice srcSlice(buffInfo_.hcclBuff.addr, hcclBuffOffset, normalSliceSize, normalSliceSize / DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, normalSliceSize, normalSliceSize / DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    CHK_RET(LocalReduce(templateResource.threads[0], srcSlice, dstSlice, param.DataDes.dataType, param.reduceType));
  
    return HcclResult::HCCL_SUCCESS;
}
 
u64 CcuTempReduceScatterMeshMem2Mem1D2Die::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl