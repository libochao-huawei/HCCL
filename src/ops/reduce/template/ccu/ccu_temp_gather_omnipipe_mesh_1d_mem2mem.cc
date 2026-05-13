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
#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2mem.h"

namespace ops_hccl {

CcuTempGatherOmniPipeMesh1DMem2Mem::CcuTempGatherOmniPipeMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                        const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }

    auto rootIt = std::find(ranks.begin(), ranks.end(), param.root);
    if (rootIt != ranks.end()) {
        mySubCommRoot_ = std::distance(ranks.begin(), rootIt);
    }

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem] mySubCommRank_=%u, mySubCommRoot_=%u, rankId=%u",
               mySubCommRank_, mySubCommRoot_, rankId);
}

CcuTempGatherOmniPipeMesh1DMem2Mem::~CcuTempGatherOmniPipeMesh1DMem2Mem()
{
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param,
                                                        const TopoInfoWithNetLayerDetails* topoInfo,
                                                        AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg& arg) {
                             return std::make_unique<CcuKernelGatherOmniPipeMesh1DMem2Mem>(arg);
                         };

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
                HCCL_ERROR("[CcuTempGatherOmniPipeMesh1DMem2Mem][CalcRes] channelProtocol=%u", channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgGatherOmniPipeMesh1DMem2Mem>(
        subCommRanks_[0].size(), mySubCommRank_, mySubCommRoot_, param, subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::FastLaunch(const OpParam& param,
                                                           const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempGatherOmniPipeMesh1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::FastLaunch] start");
    const uint64_t* args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    buffInfo_ = tempFastLaunchCtx.buffInfo;

    CcuTaskArgGatherOmniPipeMesh1DMem2Mem taskArg(
        PointerToAddr(buffInfo_.inputPtr) + args[0],
        PointerToAddr(buffInfo_.outputPtr) + args[1],
        args[2], args[3], args[4], args[5], args[6]);

    void* taskArgPtr = static_cast<void*>(&taskArg);

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0],
        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                          const TemplateDataParams& templateDataParams,
                                                          TemplateResource& templateResource)
{
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start");
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    if (localCopyFlag == 1) {
        uint64_t sliceStride = templateDataParams.inputSliceStride;
        uint64_t sliceSize = templateDataParams.sliceSize;
        uint64_t inputOmniPipeSliceStride = 0;

        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgGatherOmniPipeMesh1DMem2Mem>(
            inputAddr, outputAddr, token, sliceSize, sliceStride, localCopyFlag, inputOmniPipeSliceStride);

        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
            templateResource.ccuKernels[0], taskArgPtr));

        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] localCopy inputAddr=%llu outputAddr=%llu "
                   "sliceSize=%llu localCopyFlag=%llu",
                   inputAddr, outputAddr, sliceSize, localCopyFlag);

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[0];
        CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, token,
                               sliceSize, sliceStride, localCopyFlag, inputOmniPipeSliceStride));
        templateResource.submitInfos.push_back(submitInfo);
    } else {
        uint64_t sliceStride = stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] repeatNum=%u", repeatNum);

        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            if (sliceSize == 0) {
                continue;
            }

            uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];

            std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgGatherOmniPipeMesh1DMem2Mem>(
                inputAddr, outputAddr, token, sliceSize, sliceStride, localCopyFlag, inputOmniPipeSliceStride);

            void* taskArgPtr = static_cast<void*>(taskArg.get());
            CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
                templateResource.ccuKernels[0], taskArgPtr));

            HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] rpt=%u inputAddr=%llu outputAddr=%llu "
                       "sliceSize=%llu sliceStride=%llu localCopyFlag=%llu",
                       rpt, inputAddr, outputAddr, sliceSize, sliceStride, localCopyFlag);
        }

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[0];
        submitInfo.cachedArgs[0] = buffInfo_.inBuffBaseOff;
        submitInfo.cachedArgs[1] = buffInfo_.outBuffBaseOff;
        submitInfo.cachedArgs[2] = token;
        templateResource.submitInfos.push_back(submitInfo);
    }

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempGatherOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempGatherOmniPipeMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl