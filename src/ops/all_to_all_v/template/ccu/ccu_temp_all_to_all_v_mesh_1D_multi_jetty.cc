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
#include "ccu_kernel_all_to_all_v_mesh1d_multi_jetty.h"
#include "ccu_temp_all_to_all_v_mesh_1D_multi_jetty.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
constexpr uint32_t CONST_1 = 1;
constexpr uint32_t CONST_4 = 4;

CcuTempAllToAllVMesh1DMultiJetty::CcuTempAllToAllVMesh1DMultiJetty(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    myRank_ = rankId;
    templateRankSize_ = subCommRanks[0].size();
}

CcuTempAllToAllVMesh1DMultiJetty::~CcuTempAllToAllVMesh1DMultiJetty()
{
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllToAllVMesh1DMultiJetty>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    // 计算建链诉求以COMM_TOPO_1DMESH为优先级，优先建mesh链，没有mesh链建clos链
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
    kernelInfo.channels = channelDescs;

    std::vector<uint32_t> jettyNums;
    CHK_RET(SetJettyNums(jettyNums, false));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1DMultiJetty>(subCommRanks_[0].size(),
                                                                                    myRank_,
                                                                                    param,
                                                                                    subCommRanks_,
                                                                                    jettyNums);
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t srcOffset = 0; // alltoallv假设src起始地址为发送rank的对应块起始地址
    uint64_t dstOffset = 0; // alltoallv假设dst起始地址为接收rank的对应块起始地址

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllVMesh1DMultiJetty>(
        inputAddr, outputAddr, token, srcOffset, dstOffset, localSendRecvInfo_);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
    //所有task下发完再保存参数信息
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff,
        token, srcOffset, dstOffset, templateRankSize_));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

// executor在orchestra中调用
void CcuTempAllToAllVMesh1DMultiJetty::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::FastLaunch(const OpParam& param,
    const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] start");
    const uint64_t *args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    uint64_t rankSize = args[5];
    HcclDataType dataType = param.all2AllVDataDes.sendType;
    uint64_t dataTypeSize = SIZE_TABLE[dataType];
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] param.varMemSize [%llu] is invalid", param.varMemSize), HCCL_E_PARA);

    A2ASendRecvInfo localSendRecvInfo;
    localSendRecvInfo.sendCounts.resize(rankSize, 0);
    localSendRecvInfo.sendDispls.resize(rankSize, 0);
    localSendRecvInfo.sendLength.resize(rankSize, 0);
    localSendRecvInfo.sendOffset.resize(rankSize, 0);
    localSendRecvInfo.recvCounts.resize(rankSize, 0);
    localSendRecvInfo.recvDispls.resize(rankSize, 0);
    localSendRecvInfo.recvLength.resize(rankSize, 0);
    localSendRecvInfo.recvOffset.resize(rankSize, 0);
    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize; i++) {
        u64 val = i / rankSize;
        u64 curRank = i % rankSize;
        switch(val) {
            case 0:
                localSendRecvInfo.sendLength[curRank] = data[i] * dataTypeSize;
                break;
            case 2:
                localSendRecvInfo.sendOffset[curRank] = data[i] * dataTypeSize;
                break;
            case 3:
                localSendRecvInfo.recvOffset[curRank] = data[i] * dataTypeSize;
                break;
            default:
                break;
        }
    }
    uint64_t inputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[0];
    uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[1];

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllVMesh1DMultiJetty>(
        inputAddr, outputAddr, args[2], args[3], args[4], localSendRecvInfo);
    HCCL_INFO("[CcuTempAlltoAllVMesh1DMultiJetty::FastLaunch]: inputAddr[%llu], outputAddr[%llu], ",
        "srcOffset[%llu], dstOffset[%llu]", inputAddr, outputAddr, args[3], args[4]);
    void* taskArgPtr = static_cast<void*>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0], tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));
    HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllToAllVMesh1DMultiJetty::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::SetJettyNums(std::vector<uint32_t>& jettyNums, const bool multijetty) const
{
    jettyNums.resize(templateRankSize_, 0);
    for (int i = 0; i < templateRankSize_; i++) {
        if (i == myRank_) {
            jettyNums[i] = CONST_1;
        } else if (multijetty) {
            jettyNums[i] = CONST_4;
        } else {
            jettyNums[i] = CONST_1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl