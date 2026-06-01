/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_reduce_mesh_1D_2die_oneshot.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_kernel_all_reduce_mesh_1D_2die_oneshot.h"
#include "ccu_launch_dl.h"
#include "channel.h"

namespace ops_hccl {
constexpr u32 ALL_REDUCE_DIE_NUM = 2;
constexpr uint32_t TASK_ARG_NUM = 20;
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
        HCCL_ERROR("[CcuTempAllreduceMesh1D2DieOneShot] task arg num[%llu] is invalid.",
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

CcuTempAllreduceMesh1D2DieOneShot::CcuTempAllreduceMesh1D2DieOneShot(const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
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

CcuTempAllreduceMesh1D2DieOneShot::~CcuTempAllreduceMesh1D2DieOneShot()
{
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.ccuKernelNum.push_back(ALL_REDUCE_DIE_NUM);
    HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    std::vector<HcclChannelDesc> channelDescs;
    std::vector<HcclChannelDesc> myChannelDescs;
    std::vector<std::vector<HcclChannelDesc>> channelDescsDie(resourceRequest.ccuKernelNum[0]);
    std::vector<std::vector<u32>> groupRanksforDie(resourceRequest.ccuKernelNum[0]);

    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    for (auto channel : channelDescs) {
        if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
            myChannelDescs.push_back(channel);
        }
    }
    uint32_t channelIdx = 0;
    for (auto channel : myChannelDescs) {
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        CHK_PRT_RET(dieId >= ALL_REDUCE_DIE_NUM,
            HCCL_ERROR("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] dieId[%u] is invalid.", dieId),
            HCCL_E_INTERNAL);
        channelDescsDie[dieId].push_back(channel);
        groupRanksforDie[dieId].push_back(channel.remoteRank);
        HCCL_INFO("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] dieId[%u], channelIdx[%u]",
            dieId, channelIdx);
        channelIdx++;
    }

    for (uint32_t die = 0; die < ALL_REDUCE_DIE_NUM; die++) {
        CcuKernelInfo kernelInfo;
        strcpy(kernelInfo.kernelFuncName, "CcuAllreduceMesh1D2DieOneShotKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllreduceMesh1D2DieOneShotKernel);

        bool rmtReduceWithMyRank = channelDescsDie[die].size() <= channelDescsDie[1 - die].size();
        if (rmtReduceWithMyRank) {
            groupRanksforDie[die].push_back(mySubCommRank_);
        }

        auto kernelArg = std::make_shared<CcuKernelArgAllreduceMesh1D2DieOneShot>();
        kernelArg->dimSize_ = groupRanksforDie[die].size();
        kernelArg->rankId_ = mySubCommRank_;
        kernelArg->opParam_ = param;
        kernelArg->subCommRanks_ = groupRanksforDie;
        kernelArg->rmtReduceWithMyRank_ = rmtReduceWithMyRank;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channelDescsDie[die];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] die[%u], channelDescs.size()=%llu, dimsize=%llu",
            die, channelDescsDie[die].size(), groupRanksforDie[die].size());
    }

    HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] ccuKernelInfos.size()=%llu",
        resourceRequest.ccuKernelInfos.size());
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::KernelRun(const OpParam& param,
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

    uint32_t dataTypeSize = DataTypeSizeGet(dataType_);
    uint64_t scratchBaseOffset0 = 0;
    uint64_t scratchBaseOffset1 = sliceSize;
    uint64_t localReduceSize0 = ((sliceSize / dataTypeSize) / ALL_REDUCE_DIE_NUM) * dataTypeSize;
    uint64_t localReduceSize1 = sliceSize - localReduceSize0;
    uint64_t localReduceSliceOffset0 = 0;
    uint64_t localReduceSliceOffset1 = localReduceSize0;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE;
    auto rmtReduceGoSize = CalGoSize(sliceSize, config);
    auto localReduceGoSize0 = CalGoSize(localReduceSize0, config);
    auto localReduceGoSize1 = CalGoSize(localReduceSize1, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, scratchAddr,
        scratchBaseOffset0, scratchBaseOffset1, localReduceSliceOffset0, localReduceSliceOffset1};
    for (auto &goSize : {rmtReduceGoSize, localReduceGoSize0, localReduceGoSize1}) {
        taskArgs.insert(taskArgs.end(), goSize.begin(), goSize.end());
    }

    uint32_t argSize = static_cast<uint32_t>(taskArgs.size());
    for (uint32_t dieId = 0; dieId < ALL_REDUCE_DIE_NUM; dieId++) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieId],
            templateResource.ccuKernels[dieId], taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllreduceMesh1D2DieOneShot::KernelRun] kernel launch failed, ccuRet[%d].",
                launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        CcuKernelSubmitInfo submitInfo;
        CHK_RET(FillSubmitInfo(submitInfo, templateResource.ccuKernels[dieId], taskArgs,
            buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, buffInfo_.hcclBuffBaseOff));
        templateResource.submitInfos.push_back(submitInfo);
        HCCL_INFO("[CcuTempAllreduceMesh1D2DieOneShot::KernelRun] die[%u] end", dieId);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::FastLaunch(const OpParam& param,
    const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllreduceMesh1D2DieOneShot::FastLaunch] ccu kernel num is 0, just success.");
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
            HCCL_ERROR("[CcuTempAllreduceMesh1D2DieOneShot::FastLaunch] kernel launch failed, ccuRet[%d].",
                launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }
    return HCCL_SUCCESS;
}

u64 CcuTempAllreduceMesh1D2DieOneShot::GetThreadNum() const
{
    return ALL_REDUCE_DIE_NUM;
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;
    return HCCL_SUCCESS;
}

u64 CcuTempAllreduceMesh1D2DieOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return ALL_REDUCE_DIE_NUM;
}
} // namespace ops_hccl
