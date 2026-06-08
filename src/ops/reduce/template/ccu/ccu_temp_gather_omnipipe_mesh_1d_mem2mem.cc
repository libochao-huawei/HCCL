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
#include "alg_data_trans_wrapper.h" 

namespace ops_hccl {

CcuTempGatherOmniPipeMesh1DMem2Mem::CcuTempGatherOmniPipeMesh1DMem2Mem(const OpParam& param, const u32 rankId,
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
    rankId_ = rankId;
    // 子通信域的root卡号
    auto rootIt = std::find(ranks.begin(), ranks.end(), param.root);
    if (rootIt != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), rootIt);
    }

    ifRealRoot_ = (rankId == param.root);
    // HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem] mySubCommRank_=%u, subCommRootId_=%u, rankId=%u",
    //            mySubCommRank_, subCommRootId_, rankId);
}

CcuTempGatherOmniPipeMesh1DMem2Mem::~CcuTempGatherOmniPipeMesh1DMem2Mem()
{
}


u64 CcuTempGatherOmniPipeMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1); //TODO:啥意思

    return HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param,
                                                        const TopoInfoWithNetLayerDetails* topoInfo,
                                                        AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg& arg) {
                             return std::make_unique<CcuKernelGatherOmniPipeMesh1DMem2Mem>(arg);
                         };

    std::vector<HcclChannelDesc> channelDescs;

    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs)); //TODO:条件判断，函数

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgGatherOmniPipeMesh1DMem2Mem>(
        subCommRanks_[0].size(), mySubCommRank_, subCommRootId_, param, subCommRanks_, ifRealRoot_, myRank_);
    // kernelInfo.channels = channelDescs;
    // resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    std::set<uint32_t> mySet;
    std::vector<HcclChannelDesc> myChannels;
    for(HcclChannelDesc channel : channelDescs){
        // HCCL_INFO("[jjy]myRank:%d,channel.remoteRank:%d,channel.channelProtocol:%d",myRank_,channel.remoteRank,channel.channelProtocol);
        if(mySet.count(channel.remoteRank)==0){
            mySet.insert(channel.remoteRank);
            myChannels.push_back(channel);
        }
    }
    kernelInfo.channels = myChannels;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    resourceRequest.channels.push_back(channelDescs);

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}


HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                          const TemplateDataParams& templateDataParams,
                                                          TemplateResource& templateResource)
{
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start1");

    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inBuffBaseOff = buffInfo_.inBuffBaseOff;
    uint64_t outBuffBaseOff = buffInfo_.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff; //基址 + loop偏移
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff; //基址 + loop偏移
    
    // uint64_t token;
    // CHK_RET(GetToken(buffInfo_, token));
    uint64_t token = CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(buffInfo_.inputPtr), static_cast<uint64_t>(buffInfo_.inputSize));
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start2");

    if (localCopyFlag == 0) {
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start3");
        // 当前卡就是root卡，receive其他所有卡
        // 当前卡不是root卡，直接结束
        // uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
        uint64_t inputSliceStride = 0;
        uint64_t outputSliceStride = 0;
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] start5:%d",stepSliceInfo.inputOmniPipeSliceStride.size());
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        uint64_t sliceSize;
        uint64_t inputOmniPipeSliceStride;
        uint64_t outputOmniPipeSliceStride;
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] repeatNum=%u", repeatNum);
        // 遍历peer对端的卡
        auto inputOmniPipeSliceStrides = stepSliceInfo.inputOmniPipeSliceStride;
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] peerIdSize=%u", inputOmniPipeSliceStrides.size()); // 这里子通信域里的第几个卡
        // inputOmniPipeSliceStrides[peerId][rpt] 第peerId个卡要发到root的第rpt个数据片
        for (uint32_t peerId = 0; peerId < inputOmniPipeSliceStrides.size(); ++peerId) { // size其实是子通信域卡数
            // 判断是不是本端自己的卡
            HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] peerId=%u size=%d", peerId, inputOmniPipeSliceStrides[peerId].size());
            for (uint32_t rpt = 0; rpt < inputOmniPipeSliceStrides[peerId].size(); ++rpt) { // 子通信域的第peerId个卡，要发的第几个数据片
                sliceSize = stepSliceInfo.stepSliceSize[peerId][rpt];
                inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[peerId][rpt]; // 远端的输入offset
                outputOmniPipeSliceStride = stepSliceInfo.outputOmniPipeSliceStride[peerId][rpt]; // 远端的输出offset
                // if (templateDataParams.isloopOne_ == false) {
                //     inputOmniPipeSliceStride += 256;
                //     outputOmniPipeSliceStride += 256;
                // }
                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] sliceSize=%u", sliceSize);
                // 自己是逻辑root卡 && 自己不和自己通信
                bool ifNewRoot = (subRoot == mySubCommRank_ && peerId != subRoot); // 判断是不是root，需不需要做搬运  
                std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgGatherOmniPipeMesh1DMem2Mem>(
                    inputAddr, 
                    outputAddr,
                    token, 
                    localCopyFlag, 
                    sliceSize, 
                    inputSliceStride, 
                    outputSliceStride, 
                    inputOmniPipeSliceStride, 
                    outputOmniPipeSliceStride, 
                    isStepOne_, 
                    isLastStep_, 
                    ifNewRoot);
                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem] mySubCommRank_=%u, subCommRootId_=%u, rankId=%u",
               mySubCommRank_, subCommRootId_, rankId_);
                void* taskArgPtr = static_cast<void*>(taskArg.get());
                // HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] 209");
                // HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] repeatNum[%d] [%d] [%d]",repeatNum, templateResource.threads.size(),templateResource.ccuKernels.size());
                CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0],
                    templateResource.ccuKernels[0], taskArgPtr));
                if (ifNewRoot && sliceSize!=0) {
                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem] myRank[%u]  mySubCommRank_[%d]", myRank_, mySubCommRank_);
                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] rpt=%u inputAddr=%llu outputAddr=%llu  inBuffBaseOff=%llu outBuffBaseOff=%llu"
                            " sliceSize=%llu inputSliceStride=%llu localCopyFlag=%llu inputOmniPipeSliceStride=%llu outputOmniPipeSliceStride=%llu ifNewRoot=%llu isloopOne_t=%llu isStepOne_=%llu isLastStep_=%llu",
                            rpt, inputAddr, outputAddr, inBuffBaseOff, outBuffBaseOff, sliceSize, inputSliceStride, localCopyFlag, inputOmniPipeSliceStride,outputOmniPipeSliceStride, ifNewRoot, isloopOne_, isStepOne_, isLastStep_);
                
                }// HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] rpt=%u inputAddr=%llu outputAddr=%llu "
                //             "sliceSize=%llu inputSliceStride=%llu localCopyFlag=%llu",
                //             rpt, inputAddr, outputAddr, sliceSize, inputSliceStride, localCopyFlag);
            }
        }
    } 
    // if (localCopyFlag == 0) {
    //     uint64_t sliceSize;
    //     uint64_t inputOmniPipeSliceStride;
    //     uint64_t outputOmniPipeSliceStride;
    //     HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] subRoot=%u mySubCommRank_=%u", subRoot,  mySubCommRank_);
    //     bool ifNewRoot = (subRoot == mySubCommRank_ && isStepOne_);
    //     auto inputOmniPipeSliceStrides = stepSliceInfo.inputOmniPipeSliceStride;
    //      std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgGatherOmniPipeMesh1DMem2Mem>(
    //                 inputAddr, 
    //                 outputAddr,
    //                 token, 
    //                 localCopyFlag, 
    //                 256, 
    //                 0,
    //                 0,
    //                 512, 
    //                 512, 
    //                 1, 
    //                 0, 
    //                 ifNewRoot);
    //     void* taskArgPtr = static_cast<void*>(taskArg.get());
    //             CHK_RET(HcclCcuKernelLaunch(
    //                 param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
    // } 
    else if (localCopyFlag == 1) {
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);

        HCCL_DEBUG("[%s]buffInfo_.inputPtr[%u] buffInfo_.inBuffBaseOff[%llu] templateDataParams.sliceSize[%llu] templateDataParams.count[%llu]",
            __func__, PointerToAddr(buffInfo_.inputPtr), buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);

         HCCL_DEBUG("[%s]buffInfo_.outputPtr[%u] buffInfo_.outBuffBaseOff[%llu] templateDataParams.sliceSize[%llu] templateDataParams.count[%llu]",
            __func__, PointerToAddr(buffInfo_.outputPtr), buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
            
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
                   "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
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

} // namespace ops_hccl