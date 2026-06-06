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
#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2memY.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2memY.h"
#include "alg_data_trans_wrapper.h" 

namespace ops_hccl {

CcuTempGatherOmniPipeMesh1DMem2MemY::CcuTempGatherOmniPipeMesh1DMem2MemY(const OpParam& param, const u32 rankId,
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

    // 子通信域的root卡号
    auto rootIt = std::find(ranks.begin(), ranks.end(), param.root);
    subCommRootId_ = param.root / templateRankSize_;
    if (rootIt != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), rootIt);
    }
    rankId_ = rankId;
    ifRealRoot_ = (rankId == param.root);
    // HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY] mySubCommRank_=%u, subCommRootId_=%u, rankId=%u",
    //            mySubCommRank_, subCommRootId_, rankId);
}

CcuTempGatherOmniPipeMesh1DMem2MemY::~CcuTempGatherOmniPipeMesh1DMem2MemY()
{
}

u64 CcuTempGatherOmniPipeMesh1DMem2MemY::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2MemY::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1); //TODO:啥意思

    return HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2MemY::CalcRes(HcclComm comm, const OpParam& param,
                                                        const TopoInfoWithNetLayerDetails* topoInfo,
                                                        AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg& arg) {
                             return std::make_unique<CcuKernelGatherOmniPipeMesh1DMem2MemY>(arg);
                         };

    std::vector<HcclChannelDesc> channelDescs;

    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::CalcRes] Get Mesh Channel Success!");

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgGatherOmniPipeMesh1DMem2MemY>(
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

    // HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::CalcRes] channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu",
    //            channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}


HcclResult CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun(const OpParam& param,
                                                          const TemplateDataParams& templateDataParams,
                                                          TemplateResource& templateResource)
{
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun] start1");

    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inBuffBaseOff = buffInfo_.inBuffBaseOff;
    uint64_t outBuffBaseOff = buffInfo_.outBuffBaseOff;
    // uint64_t inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    // uint64_t outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    
    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token = CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(buffInfo_.inputPtr), static_cast<uint64_t>(buffInfo_.inputSize));
        
    // uint64_t token;
    // CHK_RET(GetToken(buffInfo_, token));
    
    // 第一步  2-->0 3-->1
    // 第2-n步 2-->0

    if (localCopyFlag == 0) {
        // uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        uint64_t sliceSize;
        uint64_t inputOmniPipeSliceStride;
        uint64_t outputOmniPipeSliceStride;
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun] subRoot=%u mySubCommRank_=%u", subRoot,  mySubCommRank_);
        auto inputOmniPipeSliceStrides = stepSliceInfo.inputOmniPipeSliceStride;
        // if (isStepOne_==false && isLastStep_==false) {
        //     for(int i=0;i<stepSliceInfo.inputOmniPipeSliceStride.size();i++){
        //         for(int j=0;j<stepSliceInfo.inputOmniPipeSliceStride[i].size();j++){
        //             HCCL_INFO("[zq][dataSliceLevel1]  myRank[%d][inputOmniPipeSliceStride][omniPipeSliceInfoG][%d][%d]:%d SliceSize[%d] subroot[%d] mySubCommRank_[%d]",myRank_,i,j,stepSliceInfo.inputOmniPipeSliceStride[i][j] , stepSliceInfo.inputOmniPipeSliceStride[i][j], subRoot, mySubCommRank_);
        //         } 
        //     }
        // }
        HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun] peerIdSize=%u", inputOmniPipeSliceStrides.size());
        for (uint32_t peerId = 0; peerId < inputOmniPipeSliceStrides.size(); ++peerId) {
            HCCL_DEBUG("[----------------] a=%lu b=%lu c=%lu d=%lu", buffInfo_.inBuffBaseOff,  buffInfo_.outBuffBaseOff, templateDataParams.stepSliceInfo.buffInfo.inBuffBaseOff, templateDataParams.stepSliceInfo.buffInfo.outBuffBaseOff);
            HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun] peerId=%u size=%d", peerId, inputOmniPipeSliceStrides[peerId].size());
            for (uint32_t rpt = 0; rpt < inputOmniPipeSliceStrides[peerId].size(); ++rpt) {
                sliceSize = stepSliceInfo.stepSliceSize[peerId][rpt];
                inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[peerId][rpt];
                outputOmniPipeSliceStride= stepSliceInfo.outputOmniPipeSliceStride[peerId][rpt];
                
                bool ifNewRoot = (subRoot == mySubCommRank_ && peerId != subRoot); //是不是能read的卡

                if (ifNewRoot && sliceSize!=0 ) { // 0 和 1
                    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY] subRoot[%u] mySubCommRank_[%u] peerId[%u] ifNewRoot[%u]  myRank_[%d] isStepOne[%d] rpt[%d]", subRoot, mySubCommRank_, peerId, ifNewRoot, myRank_, isStepOne_, rpt);
                    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun] rpt=%u inputAddr=%llu outputAddr=%llu  inBuffBaseOff=%llu outBuffBaseOff=%llu"
                            " sliceSize=%llu  localCopyFlag=%llu inputOmniPipeSliceStride=%llu outputOmniPipeSliceStride=%llu ifNewRoot=%llu isloopOne_t=%llu isStepOne_[%d] isLastStep_[%d] ",
                            rpt, inputAddr, outputAddr, inBuffBaseOff, outBuffBaseOff, sliceSize, localCopyFlag, inputOmniPipeSliceStride,outputOmniPipeSliceStride, ifNewRoot, isloopOne_, isStepOne_,isLastStep_);
                }    

                std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgGatherOmniPipeMesh1DMem2MemY>(
                    inputAddr, 
                    outputAddr,
                    token, 
                    localCopyFlag, 
                    sliceSize, 
                    inputOmniPipeSliceStride, 
                    outputOmniPipeSliceStride, 
                    isStepOne_, 
                    isLastStep_, 
                    ifNewRoot);

                void* taskArgPtr = static_cast<void*>(taskArg.get());
                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY] mySubCommRank_=%u, subCommRootId_=%u, rankId=%u", mySubCommRank_, subCommRootId_, rankId_);
                CHK_RET(HcclCcuKernelLaunch(
                    param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
                if (ifNewRoot) {
                HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun] sliceSize=%llu  inputOmniPipeSliceStride=%llu outputOmniPipeSliceStride=%llu isStepOne_[%d] isLastStep_[%d] ",
                            sliceSize, inputOmniPipeSliceStride,outputOmniPipeSliceStride, isStepOne_,isLastStep_);
                }
            }
        }
    }
    else if (localCopyFlag == 1) {
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
                   "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    }


    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2MemY::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempGatherOmniPipeMesh1DMem2MemY::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

} // namespace ops_hccl