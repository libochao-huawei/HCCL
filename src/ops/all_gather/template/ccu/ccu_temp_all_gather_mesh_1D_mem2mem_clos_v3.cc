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
#include "ccu_kernel_all_gather_mesh1d_mem2mem_clos_v3.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem_clos_v3.h"
#include "alg_env_config.h"

namespace ops_hccl {
constexpr uint64_t SLICE_RATIO_BASE = 100;

void CalcMainSharedSliceSize(uint64_t normalSliceSize, uint64_t dataTypeSize, uint64_t &mainSliceSize,
    uint64_t &sharedSliceSize)
{
    mainSliceSize = normalSliceSize * GetExternalInputCcuMainSharedRatio() / SLICE_RATIO_BASE;
    mainSliceSize = mainSliceSize / dataTypeSize * dataTypeSize;
    sharedSliceSize = normalSliceSize - mainSliceSize;
    if (mainSliceSize == 0 || sharedSliceSize == 0) {
        mainSliceSize = normalSliceSize;
        sharedSliceSize = 0;
    }
}

CcuTempAllGatherMesh1DMem2MemClosV3::CcuTempAllGatherMesh1DMem2MemClosV3(const OpParam& param, const u32 rankId,
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

CcuTempAllGatherMesh1DMem2MemClosV3::~CcuTempAllGatherMesh1DMem2MemClosV3()
{
}

HcclResult CcuTempAllGatherMesh1DMem2MemClosV3::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2MemClosV3::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllGatherMesh1DMem2MemClosV3>(arg);
                         };

    std::vector<HcclChannelDesc> channelDescs;
    std::vector<u32> mainChannelIdxByRank;
    std::vector<u32> sharedChannelIdxByRank;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopoClosV3(comm, param, topoInfo, subCommRanks_, channelDescs,
            CommTopo::COMM_TOPO_1DMESH, mainChannelIdxByRank, sharedChannelIdxByRank));
        for (auto channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
                HCCL_ERROR("[CcuTempAllGatherMesh1DMem2MemClosV3][CalcRes] channelProtocol: %u",
                           channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2MemClosV3::CalcRes] Get Mesh Channel Success!");

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2MemClosV3>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    param,
                                                                                    subCommRanks_,
                                                                                    mainChannelIdxByRank,
                                                                                    sharedChannelIdxByRank);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2MemClosV3::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DMem2MemClosV3::FastLaunch(const OpParam& param,
    const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2MemClosV3::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2MemClosV3::FastLaunch] start");
    const uint64_t *args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    CcuTaskArgAllGatherMesh1DMem2MemClosV3 taskArg(
        PointerToAddr(buffInfo_.inputPtr) + args[0],
        PointerToAddr(buffInfo_.outputPtr) + args[1],
        args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);

    void* taskArgPtr = static_cast<void*>(&taskArg);

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0],
        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2MemClosV3::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DMem2MemClosV3::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint32_t repeatNum          = templateDataParams.repeatNum;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize    = templateDataParams.sliceSize;
    uint64_t lastSliceSize      = templateDataParams.tailSize;
    bool inputOutputEqual =
        (inputAddr + inputSliceStride * mySubCommRank_ == outputAddr + outputSliceStride * mySubCommRank_);
    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);
    if (templateDataParams.tailSize != 0 && mySubCommRank_ == templateRankSize_ - 1) {
        normalSliceSize = templateDataParams.tailSize;
    }
    HCCL_INFO("[CcuTempAllGatherMesh1DMem2MemClosV3][KernelRun] normalSliceSize[%llu]", normalSliceSize);

    HcclDataType dataType       = param.DataDes.dataType;
    uint64_t dataTypeSize       = DataTypeSizeGet(dataType);
    CHK_PRT_RET(dataTypeSize == 0,
        HCCL_ERROR("[CcuTempAllGatherMesh1DMem2MemClosV3][KernelRun] invalid dataTypeSize for dataType[%u]", dataType),
        HcclResult::HCCL_E_PARA);

    uint64_t mainSliceSize = 0;
    uint64_t sharedSliceSize = 0;
    CalcMainSharedSliceSize(normalSliceSize, dataTypeSize, mainSliceSize, sharedSliceSize);
    HCCL_INFO("[CcuTempAllGatherMesh1DMem2MemClosV3][KernelRun] mainSliceSize[%llu], sharedSliceSize[%llu]",
              mainSliceSize, sharedSliceSize);

    uint64_t dataCount          = normalSliceSize / dataTypeSize;
    if (dataCount == 0 && lastSliceSize == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2MemClosV3] DataCount == 0 && lastSliceSize == 0, Template Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGatherMesh1DMem2MemClosV3>(
        inputAddr, outputAddr, token, inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, isInputOutputEqual, mainSliceSize, sharedSliceSize);

    void* taskArgPtr = static_cast<void*>(taskArg.get());
    HCCL_INFO("templateResource.threads.size[%zu], templateResource.ccuKernels.size[%zu]",
              templateResource.threads.size(), templateResource.ccuKernels.size());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token, inputSliceStride,
        outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize,
        isInputOutputEqual, mainSliceSize, sharedSliceSize));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2MemClosV3::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherMesh1DMem2MemClosV3::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempAllGatherMesh1DMem2MemClosV3::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllGatherMesh1DMem2MemClosV3::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl