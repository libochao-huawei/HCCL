/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_reduce_mesh_1D_mem2mem_2die_oneshot.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_kernel_all_reduce_mesh1d_mem2mem_2die_oneshot.h"
#include "ccu_launch.h"
#include "channel.h"

namespace ops_hccl {

constexpr u32 DIE_NUM = 2;
constexpr uint32_t TASK_ARG_NUM = 21;
constexpr uint32_t LOCAL_REDUCE_LOOP_NUM = 16;
constexpr uint32_t INPUT_IDX = 0;
constexpr uint32_t OUTPUT_IDX = 1;
constexpr uint32_t SCRATCH_IDX = 3;
constexpr uint32_t INPUT_OFFSET_IDX = TASK_ARG_NUM;
constexpr uint32_t OUTPUT_OFFSET_IDX = TASK_ARG_NUM + 1;
constexpr uint32_t SCRATCH_OFFSET_IDX = TASK_ARG_NUM + 2;

static HcclResult FillSubmitInfo(CcuKernelSubmitInfo &submitInfo, CcuKernelHandle kernelHandle,
    const std::vector<uint64_t> &taskArgs, uint64_t inputOffset, uint64_t outputOffset, uint64_t scratchOffset)
{
    CHK_PRT_RET(taskArgs.size() + 3 > CCU_MAX_TASK_ARG_NUM,
        HCCL_ERROR("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot] task arg num[%llu] is invalid.",
            taskArgs.size() + 3),
        HCCL_E_INTERNAL);

    submitInfo = {};
    submitInfo.kernelHandle = kernelHandle;
    for (uint32_t i = 0; i < taskArgs.size(); i++) {
        submitInfo.cachedArgs[i] = taskArgs[i];
    }
    submitInfo.cachedArgs[INPUT_OFFSET_IDX] = inputOffset;
    submitInfo.cachedArgs[OUTPUT_OFFSET_IDX] = outputOffset;
    submitInfo.cachedArgs[SCRATCH_OFFSET_IDX] = scratchOffset;
    return HCCL_SUCCESS;
}

CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CcuTempAllReduceMesh1DMem2Mem2DieOneShot(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
}

CcuTempAllReduceMesh1DMem2Mem2DieOneShot::~CcuTempAllReduceMesh1DMem2Mem2DieOneShot()
{
}

HcclResult CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    std::vector<std::vector<HcclChannelDesc>> channelsForDie(DIE_NUM);
    std::vector<std::vector<uint32_t>> kernelRanks(DIE_NUM);
    for (auto channel : channelDescs) {
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcRes] dieId[%u] is invalid.", dieId),
            HCCL_E_INTERNAL);
        channelsForDie[dieId].push_back(channel);
        kernelRanks[dieId].push_back(channel.remoteRank);
    }

    for (uint32_t die = 0; die < DIE_NUM; die++) {
        CcuKernelInfo kernelInfo;
        strcpy(kernelInfo.kernelFuncName, "CcuAllReduceMesh1DMem2Mem2DieOneShotKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllReduceMesh1DMem2Mem2DieOneShotKernel);

        bool rmtReduceWithMyRank = channelsForDie[die].size() <= channelsForDie[1 - die].size();
        auto kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot>();
        kernelArg->dimSize_ = subCommRanks_[0].size();
        kernelArg->rankId_ = mySubCommRank_;
        kernelArg->opParam_ = param;
        kernelArg->kernelRanks_ = kernelRanks[die];
        kernelArg->subCommRanks_ = subCommRanks_;
        kernelArg->rmtReduceWithMyRank_ = rmtReduceWithMyRank;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channelsForDie[die];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DMem2Mem2DieOneShot::KernelRun(const OpParam& param,
    const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    (void)param;
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t sliceSize = templateDataParams.sliceSize;

    uint64_t scratchBaseOffset0 = 0;
    uint64_t scratchBaseOffset1 = 0;
    uint32_t dataTypeSize = DataTypeSizeGet(dataType_);
    uint64_t localReduceSize0 = ((sliceSize / dataTypeSize) / DIE_NUM) * dataTypeSize;
    uint64_t localReduceSize1 = sliceSize - localReduceSize0;
    uint64_t localReduceSliceOffset0 = 0;
    uint64_t localReduceSliceOffset1 = localReduceSize0;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = LOCAL_REDUCE_LOOP_NUM;
    config.memSlice = CCU_MS_SIZE;
    auto localReduceGoSize = CalGoSize(sliceSize, config);
    auto localReduceGoSize0 = CalGoSize(localReduceSize0, config);
    auto localReduceGoSize1 = CalGoSize(localReduceSize1, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, scratchAddr, sliceSize,
        scratchBaseOffset0, scratchBaseOffset1, localReduceSliceOffset0, localReduceSliceOffset1};
    for (auto &goSize : {localReduceGoSize, localReduceGoSize0, localReduceGoSize1}) {
        taskArgs.insert(taskArgs.end(), goSize.begin(), goSize.end());
    }

    uint32_t argSize = static_cast<uint32_t>(taskArgs.size());
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieId],
            templateResource.ccuKernels[dieId], taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::KernelRun] kernel launch failed, ccuRet[%d].",
                launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        CcuKernelSubmitInfo submitInfo;
        CHK_RET(FillSubmitInfo(submitInfo, templateResource.ccuKernels[dieId], taskArgs,
            buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, buffInfo_.hcclBuffBaseOff));
        templateResource.submitInfos.push_back(submitInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DMem2Mem2DieOneShot::FastLaunch(const OpParam& param,
    const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }

    for (uint32_t kernelIdx = 0; kernelIdx < tempFastLaunchCtx.ccuKernelSubmitInfos.size(); kernelIdx++) {
        uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].cachedArgs);
        args[INPUT_IDX] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[INPUT_OFFSET_IDX];
        args[OUTPUT_IDX] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[OUTPUT_OFFSET_IDX];
        args[SCRATCH_IDX] = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr) + args[SCRATCH_OFFSET_IDX];

        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[kernelIdx],
            tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].kernelHandle,
            reinterpret_cast<void*>(args), TASK_ARG_NUM);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::FastLaunch] kernel launch failed, ccuRet[%d].",
                launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }
    return HCCL_SUCCESS;
}

u64 CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}
} // namespace ops_hccl
