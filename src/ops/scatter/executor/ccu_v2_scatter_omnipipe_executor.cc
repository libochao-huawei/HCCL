/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_v2_scatter_omnipipe_executor.h"
#include "omnipipe_data_slice_calc.h"
#include "ccu_temp_scatter_omnipipe_mesh_1D_mem2mem.h"
// #include "ccu_temp_scatter_omnipipe_nhr1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "coll_alg_v2_exec_registry.h"
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::
    CcuV2ScatterOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    auto userrank = topoInfo->userRank;
#if T_DESC("2x2用例", true)
    HCCL_DEBUG("[%s] 2x2-TestCase", __func__);
    if (userrank == 0 || userrank == 1) {
        algHierarchyInfo.infos = {{{0, 1}, {0, 1, 2, 3}}};
    } else {
        algHierarchyInfo.infos = {{{2, 3}, {0, 1, 2, 3}}};
    }
#elif T_DESC("4x2用例", false)
    HCCL_DEBUG("[%s] 4x2-TestCase", __func__);
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else {
        algHierarchyInfo.infos = {{{4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    }
#elif T_DESC("2x4用例", false)
    HCCL_DEBUG("[%s] 2x4-TestCase", __func__);
    if (userrank == 0 || userrank == 1) {
        algHierarchyInfo.infos = {{{0, 1}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else if (userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else if (userrank == 4 || userrank == 5) {
        algHierarchyInfo.infos = {{{4, 5}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else if (userrank == 6 || userrank == 7) {
        algHierarchyInfo.infos = {{{6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    }
#elif T_DESC("4x4用例", false)
    HCCL_DEBUG("[%s] 4x4-TestCase", __func__);
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15}}};
    } else if (userrank == 4 || userrank == 5 || userrank == 6 || userrank == 7) {
        algHierarchyInfo.infos = {{{4,5,6,7}, {0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15}}};
    } else if (userrank == 8 || userrank == 9 || userrank == 10 || userrank == 11) {
        algHierarchyInfo.infos = {{{8,9,10,11}, {0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15}}};
    } else if (userrank == 12 || userrank == 13 || userrank == 14 || userrank == 15) {
        algHierarchyInfo.infos = {{{12,13,14,15}, {0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15}}};
    }
#else
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
#endif
    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_INFO("[%s] myRank[%u] (%d, %d, %d) %u", __func__, topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    u64 rootx = param.root % rankSizeLevel0_;
    u64 rooty = param.root / rankSizeLevel0_;

    bool isRoot = (myRank_ == param.root);
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx) && !isRoot;
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty) && !isRoot;

    HCCL_INFO("[%s] myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));
    HCCL_DEBUG("[%s] myRank[%u] start", __func__, myRank_);

    // 重复的template构造
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    auto size = algHierarchyInfo.infos[0][1].size() / algHierarchyInfo.infos[0][0].size();
    HCCL_DEBUG("[%s] algHierarchyInfo.infos[0][1]size=%u algHierarchyInfo.infos[0][0]size=%u", __func__,
        algHierarchyInfo.infos[0][1].size(), algHierarchyInfo.infos[0][0].size());
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
    u32 index = 0;
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
    InsAlgTempLevel0 algTempLevel0(param, myRank_, subCommRanks0);
    InsAlgTempLevel1 algTempLevel1(param, myRank_, subCommRanks1);

    AlgResourceRequest resReqLevel0; // X
    CHK_RET(algTempLevel0.CalcRes(comm, param, topoInfo, resReqLevel0));
    AlgResourceRequest resReqLevel1; // Y
    CHK_RET(algTempLevel1.CalcRes(comm, param, topoInfo, resReqLevel1));

    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          resReqLevel0.ccuKernelInfos.begin(),
                                          resReqLevel0.ccuKernelInfos.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          resReqLevel1.ccuKernelInfos.begin(),
                                          resReqLevel1.ccuKernelInfos.end());

    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                        resReqLevel0.ccuKernelNum.begin(),
                                        resReqLevel0.ccuKernelNum.end());
    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                        resReqLevel1.ccuKernelNum.begin(),
                                        resReqLevel1.ccuKernelNum.end());

    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = 2;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的主thread
    resourceRequest.slaveThreadNum = resReqLevel0.slaveThreadNum + resReqLevel1.slaveThreadNum + 2;

    // 第一个template的zhuthread需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel0.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqLevel0.notifyNumPerThread.begin(),
                                              resReqLevel0.notifyNumPerThread.end());
    // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel1.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqLevel1.notifyNumPerThread.begin(),
                                              resReqLevel1.notifyNumPerThread.end());
    HCCL_DEBUG("[%s] slaveThreadNum[%u]", __func__, resourceRequest.slaveThreadNum);
    resourceRequest.channels.push_back(resReqLevel0.channels[0]);
    resourceRequest.channels.push_back(resReqLevel1.channels[0]);
    HCCL_DEBUG("[%s] myRank[%u] end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::RestoreChannelMap(
    const AlgResourceCtxSerializable &resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>> &rankIdToChannelInfo) const
{
    // TODOv 临时写法，待改进，两层边三层
    rankIdToChannelInfo.resize(3);
    for (u32 level = 0; level < resCtx.channels.size(); level++) {
        for (auto &channel: resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[level][remoteRank].push_back(channel);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_DEBUG("[%s] myRank[%u] start", __func__, myRank_);
    localThreads_ = resCtx.threads;
    HCCL_DEBUG("[%s]localThreads_ size[%u]", __func__, localThreads_.size()); // 3 main+x+y

    HCCL_DEBUG("[%s]myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        __func__, myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    CHK_RET(this->RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] executor kernel run failed", __func__, HCCL_ERROR_CODE(ret)), ret);
    HCCL_DEBUG("[%s] myRank[%u] end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GenTempAlgParamsIn2HCCLBuff(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount, const AlgResourceCtxSerializable &resCtx)
{
    tempAlgParams.count = 0;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = HcclBuffType::HCCL_BUFF;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GenTempAlgParamsIn2OUT(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount, const AlgResourceCtxSerializable &resCtx)
{
    tempAlgParams.count = 0;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.outputPtr = param.outputPtr;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    stepSliceInfo.buffInfo.outBuffType = BufferType::OUTPUT;
    stepSliceInfo.buffInfo.hcclBuffType = HcclBuffType::HCCL_BUFF;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GenTempAlgParamsHCCLBuff2OUT(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount, const AlgResourceCtxSerializable &resCtx)
{
    tempAlgParams.count = 0;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputPtr = param.outputPtr;
    stepSliceInfo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.outBuffType = BufferType::OUTPUT;
    stepSliceInfo.buffInfo.hcclBuffType = HcclBuffType::HCCL_BUFF;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

// 为模板准备资源
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTempLevel0 &algTempLevel0,
    InsAlgTempLevel1 &algTempLevel1)
{
    HCCL_DEBUG("[%s] start", __func__);
    HCCL_DEBUG("[%s] localThreads_ size[%u]", __func__, localThreads_.size());
    // 获取每个template的线程数
    u64 level0ThreadsNum = algTempLevel0.GetThreadNum();
    u64 level1ThreadsNum = algTempLevel1.GetThreadNum();
    HCCL_DEBUG("[%s]level0ThreasNum[%u] level1ThreadsNum[%u]", __func__, level0ThreadsNum, level1ThreadsNum);

    level0Threads_.assign(localThreads_.begin() + 1, localThreads_.begin() + 1 + level0ThreadsNum);
    level1Threads_.assign(localThreads_.begin() + 1 + level0ThreadsNum, localThreads_.end());
    HCCL_DEBUG("[%s]level0Threads size[%u] level1Threads size[%u]",
        __func__, level0Threads_.size(), level1Threads_.size());

    // 控制线程用于算法同步
    controlThread_ = localThreads_.at(0);
    // xy轴各自的主线程
    templateMainThreads_.push_back(level0Threads_.at(0));
    templateMainThreads_.push_back(level1Threads_.at(0));
    HCCL_DEBUG("[%s]templateMainThreads size[%u]", __func__, templateMainThreads_.size());

    // 获取template各自的主thread上有多少notify
    AlgResourceRequest level0TempRequest;
    CHK_RET(algTempLevel0.GetRes(level0TempRequest));
    notifyIdxControlToTemplates_.push_back(level0TempRequest.notifyNumOnMainThread);
    AlgResourceRequest level1TempRequest;
    CHK_RET(algTempLevel1.GetRes(level1TempRequest));
    notifyIdxControlToTemplates_.push_back(level1TempRequest.notifyNumOnMainThread);
    notifyIdxTemplatesToControl_.push_back(0);
    notifyIdxTemplatesToControl_.push_back(1);
    HCCL_DEBUG("[%s]notifyIdxControlToTemplates_ size[%u]", __func__, notifyIdxControlToTemplates_.size());
    HCCL_DEBUG("[%s]notifyIdxTemplatesToControl_ size[%u]", __func__, notifyIdxTemplatesToControl_.size());

    // 单独本地拷贝使用
    templateLocalCopyThreads_.push_back(level0Threads_.at(0));

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

// 主执行函数
template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::OrchestrateLoop(const OpParam &param, 
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[%s] Start", __func__);
    auto algHierarchyInfo = resCtx.algHierarchyInfo;
    bool isRoot = (myRank_ == param.root);
    
    // 构造subCommRanks
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    auto size = algHierarchyInfo.infos[0][1].size() / algHierarchyInfo.infos[0][0].size();
    // std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
    std::vector<std::vector<u32>> subCommRanks1;
    subCommRanks1.resize(1);
    for (int i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0].push_back(algHierarchyInfo.infos[0][1][i]);
        HCCL_DEBUG("subCommRanks1 localRank[%u] push_back[%u]", myRank_, resCtx.algHierarchyInfo.infos[0][1][i]);
    }

    // 打印子通信组信息
    for (size_t i = 0; i < subCommRanks0.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks0[i].size(); ++j) {
            ss << subCommRanks0[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks0[%zu] content: %s", __func__, i, ss.str().c_str());
    }

    for (size_t i = 0; i < subCommRanks1.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks1[i].size(); ++j) {
            ss << subCommRanks1[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks1[%zu] content: %s", __func__, i, ss.str().c_str());
    }
    
    // 创建template实例
    InsAlgTempLevel0 algTempX(param, myRank_, subCommRanks0);
    InsAlgTempLevel1 algTempY(param, myRank_, subCommRanks1);
    
    // 公共参数初始化
    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.inputSize = param.inputSize;
    tempAlgParamsCommon.buffInfo.outputSize = param.outputSize;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.inputSliceStride = dataSize_;
    tempAlgParamsCommon.outputSliceStride = dataSize_;
    
    // 资源模板初始化
    TemplateResource templateResourceCommon;
    templateResourceCommon.threads = resCtx.threads;
    
    // 通道处理
    TemplateResource templateResourceX = templateResourceCommon;
    // templateResourceX.threads = level0Threads_;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateResourceX.channels = remoteRankToChannelInfo_[0];
    }
    
    TemplateResource templateResourceY = templateResourceCommon;
    // templateResourceY.threads = level1Threads_;
    if (remoteRankToChannelInfo_.size() > 1) {
        templateResourceY.channels = remoteRankToChannelInfo_[1];
    }
    
    // 为template准备资源
    CHK_RET(PrepareResForTemplate(param, resCtx, algTempX, algTempY));
    // 1、计算带宽 平均带宽还是总带宽,如果是总带宽这边要处理成平均带宽
    std::vector<std::vector<EndpointAttrBwCoeff>> endpointAttrBw;
    std::vector<EndpointAttrBwCoeff> endpointAttrBwAvg;
#if T_DESC("计算带宽实现", false)
    CHK_RET(CalAllLevelEndpointAttrBwCoeff(param.hcclComm, myRank_, 3, endpointAttrBw));
    // 需要转化成平均带宽
    u64 bwIndex = 0;
    for (u64 i = 0; i < endpointAttrBw.size(); i++) {
        for (u64 j = 0; j < endpointAttrBw[i].size(); ++j) {
            endpointAttrBw[i][j] /= algHierarchyInfo.infos[i][j].size() - 1;
            endpointAttrBwAvg[bwIndex++] = endpointAttrBw[i][j];
        }
    }
#else
    endpointAttrBwAvg = {3,4,1};
#endif

    // 计算loop相关信息
    u64 maxCountPerLoop = static_cast<u64>(256) / dataTypeSize_;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%u]", __func__, myRank_, loopTimes);
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s] perLoopSize[%u]", __func__, perLoopSize);

#if 0
    // 2、计算scratch 返回的数组0是maxCountPerloop, 1是loopTimes
    OmniPipeScratchParam scratchParam;
    scratchParam.endpointAttrBw = endpointAttrBwAvg;
    scratchParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_};
    scratchParam.levelAlgType = {1,0,1};
    // 手动转成数组，这边只给reducescatter用
    std::vector<u64> dataSizeVec;
    for (int i = 0; i < rankSize_; i++) {
        dataSizeVec.push_back(dataSize_);
    }
    // TODO:调测的时候可以按照reducescatter的方式
    scratchParam.dataSize = dataSizeVec;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam);
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];
    HCCL_DEBUG("loopInfo[0](maxCountPerLoop)=%u, loopInfo[1](loopTimes)=%u", maxCountPerLoop, loopTimes)
#endif
    
    // 计算对齐数据的切片信息
    OmniPipeSliceInfo alignSliceInfo;
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, dataSize_);
    
    // 填充参数
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    sliceParam.endpointAttrBw = endpointAttrBwAvg; // 默认带宽系数
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0}; // z轴默认为0
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1}; // z轴默认为1
    std::vector<u64> levelAlgType{1, 0, 1}; // MESH算法
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    
    alignSliceInfo = CalcScatterOmniPipeSliceInfo(sliceParam, param.root);
    
    // 计算尾数据的切片信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        // 计算每个rank的数据量
        u64 tailCount = dataCount_ % maxCountPerLoop;
        std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, tailCount * dataTypeSize_);
        
        sliceParam.dataSizePerLoop = dataSizePerLoop;
        sliceParam.dataWholeSize = dataWholeSize;
        
        tailSliceInfo = CalcScatterOmniPipeSliceInfo(sliceParam, param.root);
    }
    
    // 处理所有loop
    u64 processedDataCount = 0;
    TemplateDataParams tempAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempAlgParamsY = tempAlgParamsCommon;
    // 确定当前使用的切片信息
    OmniPipeSliceInfo currentSliceInfo;
    
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        HCCL_DEBUG("[%s] myRank[%u] currDataCount[%llu]", __func__, myRank_, currDataCount);
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            currentSliceInfo = tailSliceInfo;
        } else {
            currentSliceInfo = alignSliceInfo;
        }
        
        // 获取步骤数量
        auto innerServerStepNum = currentSliceInfo.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] innerServerStepNum[%u]", __func__, myRank_, innerServerStepNum);
        
        // 处理所有步骤
        for (auto i = 0; i < innerServerStepNum; ++i) {
            // 开始前同步
            CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
            
            // 清空之前的kernel列表
            templateResourceX.ccuKernels.clear();
            templateResourceY.ccuKernels.clear();
            
            // 添加当前步骤需要的kernel
            templateResourceX.ccuKernels.insert(templateResourceX.ccuKernels.end(),
                resCtx.ccuKernels.begin(),
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
            templateResourceY.ccuKernels.insert(templateResourceY.ccuKernels.end(),
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
            
            // 第一步：只有当前节点时root时才发送对角数据给同轴节点
            if (i == 0 && isRoot) {
                // root节点发送对角数据给x轴和y轴的同轴线节点
                // 数据流向：input --> hcclbuff
                // 生成X维度（慢轴）的template参数
                CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));

                // 执行X维度通信
                CHK_RET(algTempX.KernelRun(param, tempAlgParamsX, templateResourceX));
                
                // 生成Y维度（快轴）的template参数
                CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempAlgParamsY, currentSliceInfo.dataSliceLevel1[i], processedDataCount, resCtx, param));
                
                // 执行Y维度通信
                CHK_RET(algTempY.KernelRun(param, tempAlgParamsY, templateResourceY));
            }
            // 最后一步：多角色并行处理
            // - root节点：快轴发全部同轴数据，慢轴发剩余同轴数据
            // - 同x轴非root节点：往y轴方向转发数据（NHR算法）
            // - 同y轴非root节点：往x轴方向转发数据（mesh算法）
            else if (i == innerServerStepNum - 1) {
                // root节点：快轴(Y轴)发全部同轴数据，慢轴(x轴)发剩余同轴数据
                // 数据流向：发同轴数据时时input --> output
                if (isRoot) {
                    CHK_RET(GenTempAlgParamsIn2OUT(tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    CHK_RET(algTempX.KernelRun(param, tempAlgParamsX, templateResourceX));
                    CHK_RET(GenTempAlgParamsIn2OUT(tempAlgParamsY, currentSliceInfo.dataSliceLevel1[i], processedDataCount, resCtx, param)); 
                    CHK_RET(algTempY.KernelRun(param, tempAlgParamsY, templateResourceY));
                }
                
                // 同慢轴(x轴)的非root节点：沿y轴方向发送转发数据（使用NHR算法,走templateY）
                // 数据流向：hcclbuff --> output（使用NHR算法）
                if (isSameXAxisAsRoot) {
                    // u32 newRootX = root_ / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_;
                    algTempY.SetRoot(myRank_);
                    CHK_RET(GenTempAlgParamsHCCLBuff2OUT(tempAlgParamsY, currentSliceInfo.dataSliceLevel1[i], processedDataCount, resCtx, param));
                    CHK_RET(algTempY.KernelRun(param, tempAlgParamsY, templateResourceY));
                }
                
                // 同快轴(y轴)的非root节点：沿x轴方向发送转发数据（使用mesh算法,走templateX）
                // 数据流向：hcclbuff --> output
                if (isSameYAxisAsRoot) {
                    // u32 newRootY = root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankSizeLevel0_;
                    algTempX.SetRoot(myRank_);
                    CHK_RET(GenTempAlgParamsHCCLBuff2OUT(tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    CHK_RET(algTempX.KernelRun(param, tempAlgParamsX, templateResourceX));
                }
            }
            // 中间步骤：逐步转发阶段
            // - root节点：快轴发对角数据，慢轴发同轴数据
            // - 同x轴非root节点：不发送任何数据（等待接收）
            // - 同y轴非root节点：往x轴方向发送部分转发数据
            else {
                // root节点：快轴(Y轴)发对角数据(input->hcclbuff)，慢轴(x轴)发同轴数据(input-->output)
                if (isRoot) {
                    // 数据流向：input-->output
                    CHK_RET(GenTempAlgParamsIn2OUT(tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    CHK_RET(algTempX.KernelRun(param, tempAlgParamsX, templateResourceX));

                    // 数据流向：input->hcclbuff
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempAlgParamsY, currentSliceInfo.dataSliceLevel1[i], processedDataCount, resCtx, param));
                    CHK_RET(algTempY.KernelRun(param, tempAlgParamsY, templateResourceY));
                }
                
                // 同慢轴(x轴)的非root节点：此阶段不发送任何数据，只接收
                // if (isSameXAxisAsRoot) { }  // 空分支，显式说明不发送
                
                // 同快轴(y轴)的非root节点：往x轴方向发送部分转发数据（使用mesh算法,走templateX）
                // 数据流向：hcclbuff --> output
                if (isSameYAxisAsRoot) {
                    // u32 newRootY = root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankSizeLevel0_;
                    algTempX.SetRoot(myRank_);
                    CHK_RET(GenTempAlgParamsHCCLBuff2OUT(tempAlgParamsX, currentSliceInfo.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    CHK_RET(algTempX.KernelRun(param, tempAlgParamsX, templateResourceX));
                }
            }
            // 步骤完成后同步
            CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));
        }
        
        processedDataCount += currDataCount;
        HCCL_DEBUG("[%s] myRank[%u] processedDataCount[%llu]", __func__, myRank_, processedDataCount);
    }

    HCCL_INFO("[%s] End", __func__);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::GetRes(
    AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 2;
    resourceRequest.notifyNumOnMainThread = 2;
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_SCATTER, CcuV2ScatterOmniPipe, CcuV2ScatterOmniPipeExecutor, TopoMatchUBX, CcuTempScatterOmniPipeMesh1DMem2Mem, CcuTempScatterOmniPipeMesh1DMem2Mem);

} // namespace ops_hccl
