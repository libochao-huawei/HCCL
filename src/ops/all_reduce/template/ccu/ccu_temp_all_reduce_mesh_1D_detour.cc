/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_reduce_mesh_1D_detour.h"
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "ccu_kernel_all_reduce_mesh1d_detour.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

constexpr uint64_t MS_SIZE = 4096;

CcuTempAllReduceMesh1DDetour::CcuTempAllReduceMesh1DDetour(const OpParam& param, 
                                                const u32 rankId,
                                                const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    templateRankSize_ = subCommRanks[0].size();
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
}

CcuTempAllReduceMesh1DDetour::~CcuTempAllReduceMesh1DDetour()
{
}

void CcuTempAllReduceMesh1DDetour::CalcDetourOffset(uint64_t sliceSize, uint64_t &tailOffset, uint64_t &tailSize, uint64_t &iterNum)
{
    uint64_t loopSize = pathNumPerPeer_ * MS_SIZE * CcuRep::CCU_MS_DEFAULT_LOOP_COUNT;  // 整块迭代
    CHK_PRT_RET(loopSize == 0, HCCL_ERROR("loopSize is 0."), );
    tailSize = sliceSize % loopSize;
    tailOffset = sliceSize - tailSize;
    iterNum = sliceSize / loopSize;
    return;
}

HcclResult CcuTempAllReduceMesh1DDetour::CalcKernelArgs()
{
    for (u32 r = 0; r < templateRankSize_ - 1; r++) {
        u32 neighborRank = subCommRanks_[0][(mySubCommRank_ + 1 + r) % templateRankSize_];
        if (rankIdToChannelDesc_.count(neighborRank) == 0) {
            HCCL_ERROR("[CalcKernelArgs] neighborRank[%u] not in rankIdToChannelDesc", neighborRank);
            return HCCL_E_INTERNAL;
        }
        u32 linkNum = rankIdToChannelDesc_.at(neighborRank).size();
        // 2P支持2,3,4条link，4P支持2条link，注意绕路link分两条
        CHK_PRT_RET((templateRankSize_ == 2 && (linkNum <= 1 || linkNum > 1 + 3 * 2)) ||
                    (templateRankSize_ == 4 && linkNum != 1 + 1 * 2),// 4P场景下，1条直连，绕路拆成2条
            HCCL_ERROR("[CalcKernelArgs] Invalid linkNum[%u] for RankSize[%u].", linkNum, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);
        if (r == 0) {
            detourPathNum_ = (templateRankSize_ == 2) ? (linkNum - 1) / 2 : 1; // 2P时去掉直连有2N条绕路link，对应N个绕路路径
            pathNumPerPeer_ = (templateRankSize_ == 2) ? (detourPathNum_ + 1) : detourPathNum_ + 2;  // 4P直连有2条，固定3条
        }
    }

    singleTransferSize_ = 0;
    lengths_.clear(); //多轮情况下每轮需要清零
    for (u32 i = 0; i < pathNumPerPeer_; i++) {
        lengths_.emplace_back(MS_SIZE);
        singleTransferSize_ += MS_SIZE;
    }
    HCCL_INFO("[CalcKernelArgs] detourPathNum[%u], pathNum[%u], singleTransferSize[%llu], lengths.size()=[%llu]",
              detourPathNum_, pathNumPerPeer_, singleTransferSize_, lengths_.size());
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DDetour::CalcSliceInfo(const u64 dataSize, RankSliceInfo &sliceInfoVec)
{
    CHK_RET(CalcSliceInfoAllReduce(templateRankSize_, dataSize, sliceInfoVec)); // ***
    return HcclResult::HCCL_SUCCESS;
}

/*
 * 一个基本的 Allreduce 数据切分函数，用于ReduceScatter + Allgather组合成的 Allreduce 算。
 * 输入的 dataSize 是一张卡上完整的数据量
 * 函数会将 dataSize 切分成 rankSize 份，最后一份尾块可能会比其他的切分出来的子块大。
 */
HcclResult CcuTempAllReduceMesh1DDetour::CalcSliceInfoAllReduce(const u32 rankSize, const u64 dataSize, RankSliceInfo &sliceInfoVec)
{
    sliceInfoVec.clear();
    sliceInfoVec.resize(rankSize);

    u64 unitAllignSize = DataTypeSizeGet(dataType_);
    u64 unitPerSlice = dataSize / unitAllignSize / rankSize;
    HCCL_DEBUG("unitAllignSize[%llu] unitPerSlice[%llu]", unitAllignSize, unitPerSlice);

    u64       accumOff = 0;
    SliceInfo currSlice;
    for (u32 rankIdx = 0; rankIdx < rankSize; rankIdx++) {
        if (rankIdx == rankSize - 1) {
            currSlice.offset = accumOff;
            currSlice.size   = dataSize - accumOff;
        } else {
            currSlice.offset = accumOff;
            currSlice.size   = unitPerSlice * unitAllignSize;
        }
        CHK_PRT_RET(currSlice.size % unitAllignSize != 0,
                    HCCL_ERROR("[Calc][SliceInfo]rank[%u] slice size[%llu] is invalid, unitAllignSize[%llu]",
                               rankIdx, currSlice.size, unitAllignSize),
                    HcclResult::HCCL_E_INTERNAL);
        sliceInfoVec[rankIdx].push_back(currSlice);
        accumOff += currSlice.size;
    }

    CHK_PRT_RET((sliceInfoVec[rankSize - 1][0].offset + sliceInfoVec[rankSize - 1][0].size != dataSize),
                HCCL_ERROR("[CalcSliceInfoAllReduce] SliceInfo calculation error! DataSize[%llu], "
                           "lastoffset[%llu], lastsize[%llu]",
                           dataSize, sliceInfoVec[rankSize - 1][0].offset, sliceInfoVec[rankSize - 1][0].size),
                HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DDetour::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{   
    // 当前仅支持2P或4P
    CHK_PRT_RET(templateRankSize_ != 2 && templateRankSize_ != 4,
        HCCL_INFO("[CcuTempAllReduceMeshDetour1D] Invalid RankSize[%u].", templateRankSize_), HcclResult::HCCL_E_INTERNAL);
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<u32> channelsIndexVec;
    CHK_RET(CalcChannelRequestMesh1DDetour(comm, param, topoInfo, subCommRanks_, channelDescs, channelsIndexVec));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    CHK_RET(CalcKernelArgs());
    
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllReduceMesh1DDetour::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllReduceMesh1DDetour>(arg);
                         };

    std::vector<uint64_t> dimSize;
    dimSize.emplace_back(subCommRanks_[0].size());
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1DDetour>(dimSize,
                                                                            mySubCommRank_,
                                                                            param,
                                                                            subCommRanks_,
                                                                            singleTransferSize_,
                                                                            detourPathNum_,
                                                                            pathNumPerPeer_,
                                                                            channelsIndexVec);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMesh1D::CalcRes] channelDescs.size()=%llu, subCommRanks.size()=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1DDetour::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                   const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSliceInfo(templateDataParams.sliceSize, sliceInfoVec));
    CHK_PTR_NULL(buffInfo_.inputPtr);
    CHK_PTR_NULL(buffInfo_.outputPtr);

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));  

    uint64_t sliceSize = sliceInfoVec[myRank_][0].size;  // 获取本rank需要处理的数据量
    uint64_t offSet = sliceInfoVec[myRank_][0].offset;   // 自己需要 reduce 的数据基于 inputAddr 的偏移
    uint64_t tailOffset;
    uint64_t tailSize;
    uint64_t iterNum;
    CalcDetourOffset(sliceSize, tailOffset, tailSize, iterNum);

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceMesh1DDetour>(inputAddr, outputAddr, offSet, token, iterNum, 
        tailOffset, tailSize, lengths_);
    HCCL_INFO("[CcuTempAllReduceMeshDetour1D] KernelRun: myRank_[%d], inputAddr[%llu], outputAddr[%llu],"\
        "sliceSize[%llu], offset[%llu], iterNum[%llu], tailOffset[%llu], tailSize[%llu], lengths_.size[%llu]",
        myRank_, inputAddr, outputAddr, sliceSize, offSet, iterNum, tailOffset, tailSize, lengths_.size());

    void* taskArgPtr = static_cast<void*>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
    return HcclResult::HCCL_SUCCESS;
}

}
