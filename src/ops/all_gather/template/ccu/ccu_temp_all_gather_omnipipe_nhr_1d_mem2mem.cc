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
#include "alg_data_trans_wrapper.h"
#include "ccu_kernel_all_gather_omnipipe_nhr_1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_nhr_1d_mem2mem.h"

namespace ops_hccl {

CcuTempAllGatherOmniPipeNHR1DMem2Mem::CcuTempAllGatherOmniPipeNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
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

CcuTempAllGatherOmniPipeNHR1DMem2Mem::~CcuTempAllGatherOmniPipeNHR1DMem2Mem()
{
}


HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum)
{
    constexpr u32 LINK_NUM_1 = 2;
    constexpr u32 LINK_NUM_2 = 2;
    auto firstElement = rankIdToChannelDesc_.begin();
    const std::vector<HcclChannelDesc>& firstVector = firstElement->second;
    if (firstVector.size() == 1) {
        dieNum = 1;
        return HcclResult::HCCL_SUCCESS;
    } else if (firstVector.size() == LINK_NUM_2) {
        // 检查2个channel是否在2个die上
        uint32_t dieId0 = 0;
        uint32_t dieId1 = 0;
        GetChannelDieId(comm, myRank_, firstVector[0], dieId0);
        GetChannelDieId(comm, myRank_, firstVector[1], dieId1);
        if (dieId0 == dieId1) {
            dieNum = LINK_NUM_1;
        } else {
            dieNum = LINK_NUM_2;
        }
        return HcclResult::HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcRes] get channelDescs fail: there are [] link to rank []",
                   firstVector.size(), firstElement->first);
        return HcclResult::HCCL_E_INTERNAL;
    }
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::ProcessNHRStepInfo(HcclComm comm, const std::vector<HcclChannelDesc>& channelDescs,
                                                            std::vector<NHRStepInfo>& stepInfoVector,
                                                            std::map<u32, u32>& rank2ChannelIdx)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (rank2ChannelIdx.count(stepInfo.fromRank) == 0) {
            // 存储 rankid → channelIdx 的索引
            for (int i = 0; i < channelDescs.size(); i++) {
                if (channelDescs[i].remoteRank == stepInfo.fromRank) {
                    rank2ChannelIdx[stepInfo.fromRank] = i;
                }
            }
        }
        if (rank2ChannelIdx.count(stepInfo.toRank) == 0) {
            // 存储 rankid → channelIdx 的索引
            for (int i = 0; i < channelDescs.size(); i++) {
                if (channelDescs[i].remoteRank == stepInfo.toRank) {
                    rank2ChannelIdx[stepInfo.toRank] = i;
                }
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                         AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    resourceRequest.ccuKernelNum.push_back(threadNum);
    HCCL_DEBUG("[CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;
    CHK_RET(ProcessNHRStepInfo(comm, channelDescs, stepInfoVector, rank2ChannelIdx));

    // 3.构造kernelInfo
    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                            return std::make_unique<CcuKernelAllGatherOmniPipeNHR1DMem2Mem>(arg);
                        };
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem>(subCommRanks_[0].size(),
                                                                                      mySubCommRank_,
                                                                                      stepInfoVector, rank2ChannelIdx,
                                                                                      param, subCommRanks_);

    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    resourceRequest.channels.push_back(channelDescs);

    HCCL_DEBUG("[CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcScratchSlice(u64 dataSize)
{
    // mesh直接乘rankSize
    u64 scratchMultiple = templateRankSize_ * dataSize;
    return scratchMultiple;
}

u64 CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetThreadNum()
{
    return 1;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    return HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    u32 rankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = mySubCommRank_;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    // 计算本rank在本轮收/发中的slice编号
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;

        HCCL_INFO("[AllGatherNHR1D][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherOmniPipeNHR1DMem2Mem::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllGatherOmniPipeNHR1DMem2Mem] Template KernelRun start.");

    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;

    uint32_t rankId             = myRank_;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));

    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    uint64_t sliceStride = templateDataParams.inputSliceStride;
    uint64_t sendCount = templateDataParams.count;
    uint64_t sliceSize = templateDataParams.sliceSize;

    std::vector<uint64_t> inputOmniPipeSliceStride;
    inputOmniPipeSliceStride.insert(inputOmniPipeSliceStride.begin(),
        templateDataParams.inputOmniPipeSliceStride.begin(), templateDataParams.inputOmniPipeSliceStride.end());

    std::vector<uint64_t> outputOmniPipeSliceStride;
    outputOmniPipeSliceStride.insert(outputOmniPipeSliceStride.begin(),
        templateDataParams.outputOmniPipeSliceStride.begin(), templateDataParams.outputOmniPipeSliceStride.end());

    uint64_t repeatNum = inputOmniPipeSliceStride.size();
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem>(
        inputAddr, outputAddr, token, sendCount, sliceSize, repeatNum, sliceStride, localCopyFlag,
        inputOmniPipeSliceStride, outputOmniPipeSliceStride);

    void* taskArgPtr = static_cast<void*>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    HCCL_DEBUG("[CcuTempAllGatherOmniPipeNHR1DMem2Mem::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherOmniPipeNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}
} // namespace ops_hccl