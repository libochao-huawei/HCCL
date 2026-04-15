/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_v2_reduce_scatter_omnipipe_executor.h"
#include "ccu_temp_reduce_scatter_omnipipe_mesh_1D.h"
namespace ops_hccl {

#if T_DESC("屏蔽Level1", false)
    #define TEST_NO_LEVEL1
#endif

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::
    CcuV2ReduceScatterOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::
    CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
#if T_DESC("测试拓扑2*2", true)
    HCCL_DEBUG("[%s] myRank[%u]", __func__, topoInfo->userRank);
    switch (topoInfo->userRank) {
        case 0:
        case 1:
            algHierarchyInfo.infos = {{{0, 1}, {0, 1, 2, 3}}};
            break;
        case 2:
        case 3:
            algHierarchyInfo.infos = {{{2, 3}, {0, 1, 2, 3}}};
            break;
        default:
            return HcclResult::HCCL_E_PARA;
    }

    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_INFO("[%s] myRank[%u] (%d, %d, %d) %u", __func__, topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }
#else
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
#endif
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

#ifndef TEST_NO_LEVEL1
    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 1", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel1_;
#else
    rankSizeLevel1_ = 0;
    rankIdxLevel1_ = 0;
#endif
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;

    HCCL_INFO("[%s]myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[%s] start", __func__);
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    // 重复的template构造
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
#ifndef TEST_NO_LEVEL1
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(2, 0));
    u32 index = 0;
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
#endif
    InsAlgTempLevel0 algTempLevel0(param, myRank_, subCommRanks0);
#ifndef TEST_NO_LEVEL1
    InsAlgTempLevel1 algTempLevel1(param, myRank_, subCommRanks1);
#endif

    AlgResourceRequest resReqLevel0; // X
    CHK_RET(algTempLevel0.CalcRes(comm, param, topoInfo, resReqLevel0));
#ifndef TEST_NO_LEVEL1
    AlgResourceRequest resReqLevel1; // Y
    CHK_RET(algTempLevel1.CalcRes(comm, param, topoInfo, resReqLevel1));
#endif

    resourceRequest.ccuKernelInfos.insert(
        resourceRequest.ccuKernelInfos.end(), resReqLevel0.ccuKernelInfos.begin(), resReqLevel0.ccuKernelInfos.end());
#ifndef TEST_NO_LEVEL1
    resourceRequest.ccuKernelInfos.insert(
        resourceRequest.ccuKernelInfos.end(), resReqLevel1.ccuKernelInfos.begin(), resReqLevel1.ccuKernelInfos.end());
#endif

    resourceRequest.ccuKernelNum.insert(
        resourceRequest.ccuKernelNum.end(), resReqLevel0.ccuKernelNum.begin(), resReqLevel0.ccuKernelNum.end());
#ifndef TEST_NO_LEVEL1
    resourceRequest.ccuKernelNum.insert(
        resourceRequest.ccuKernelNum.end(), resReqLevel1.ccuKernelNum.begin(), resReqLevel1.ccuKernelNum.end());
#endif

    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = 2;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的主thread
#ifndef TEST_NO_LEVEL1
    resourceRequest.slaveThreadNum = resReqLevel0.slaveThreadNum + resReqLevel1.slaveThreadNum + 2;
#else
    resourceRequest.slaveThreadNum = resReqLevel0.slaveThreadNum + 1;
#endif

    // 第一个template的zhuthread需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel0.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        resReqLevel0.notifyNumPerThread.begin(), resReqLevel0.notifyNumPerThread.end());
    // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
#ifndef TEST_NO_LEVEL1
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel1.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        resReqLevel1.notifyNumPerThread.begin(), resReqLevel1.notifyNumPerThread.end());
#endif
    HCCL_DEBUG("[%s] slaveThreadNum[%u]", __func__, resourceRequest.slaveThreadNum);
    resourceRequest.channels.push_back(resReqLevel0.channels[0]);
#ifndef TEST_NO_LEVEL1
    resourceRequest.channels.push_back(resReqLevel1.channels[0]);
#endif
    HCCL_DEBUG("[%s] End", __func__);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::RestoreChannelMap(
    const AlgResourceCtxSerializable &resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>> &rankIdToChannelInfo)
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
HcclResult CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[%s]Start", __func__);
    threads_ = resCtx.threads;
    HCCL_DEBUG("[%s]threads_ size[%u]", __func__, threads_.size()); // 3 main+x+y

    HCCL_DEBUG("[%s]myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        __func__, myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    CHK_RET(this->RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] executor kernel run failed", __func__, HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("[%s]Run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename M, typename X, typename Y>
HcclResult CcuV2ReduceScatterOmniPipeExecutor<M, X, Y>::GenTemplateAlgParamsByDimData(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount) {

    tempAlgParams.buffInfo.inBuffType = stepSliceInfo.buffInfo.inBuffType;
    tempAlgParams.buffInfo.outBuffType = stepSliceInfo.buffInfo.outBuffType;

    tempAlgParams.count = stepSliceInfo.count;  // 此斜对角step发送的数据量
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;

    HCCL_DEBUG("[%s]myRank[%u] inBuffBaseOff[%llu] processedDataCount[%llu] end inBuffBaseOff[%llu]", __func__,
        myRank_, stepSliceInfo.buffInfo.inBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.inBuffBaseOff);

    HCCL_DEBUG("[%s]myRank[%u] outBuffBaseOff[%llu] processedDataCount[%llu] end outBuffBaseOff[%llu]", __func__,
        myRank_, stepSliceInfo.buffInfo.outBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.outBuffBaseOff);

    tempAlgParams.inputSliceStride = stepSliceInfo.inputSliceStride;
    tempAlgParams.outputSliceStride = stepSliceInfo.outputSliceStride;
    tempAlgParams.sliceSize = stepSliceInfo.sliceSize;

    // 新增
    tempAlgParams.inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride;
    tempAlgParams.outputOmniPipeSliceStride = stepSliceInfo.outputOmniPipeSliceStride;
    tempAlgParams.localCopyFlag = 0;
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult
    CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::PrepareResForTemplate(
        const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTempLevel0 &algTempLevel0,
        InsAlgTempLevel1 &algTempLevel1)
{
    HCCL_DEBUG("[%s] start", __func__);
    HCCL_DEBUG("[%s] threads size[%u]", __func__, threads_.size());
    // 获取每个temp的线程数
    u64 level0ThreadsNum = algTempLevel0.GetThreadNum();
#ifndef TEST_NO_LEVEL1
    u64 level1ThreadsNum = algTempLevel1.GetThreadNum();
#else
    u64 level1ThreadsNum = 0;
#endif
    HCCL_DEBUG("[%s]level0ThreasNum[%u] level1ThreadsNum[%u]", __func__, level0ThreadsNum, level1ThreadsNum);

    level0Threads_.assign(threads_.begin() + 1, threads_.begin() + 1 + level0ThreadsNum);
    level1Threads_.assign(threads_.begin() + 1 + level0ThreadsNum, threads_.end());
    HCCL_DEBUG("[%s]level0Threads size[%u] level1Threads size[%u]",
        __func__, level0Threads_.size(), level1Threads_.size());

    // 控制线程 用于算法同步
    controlThread_ = threads_.at(0);
    // xy轴各自的主线程
    templateMainThreads_.push_back(level0Threads_.at(0));
#ifndef TEST_NO_LEVEL1
    templateMainThreads_.push_back(level1Threads_.at(0));
#endif
    HCCL_DEBUG("[%s]templateMainThreads size[%u]", __func__, templateMainThreads_.size());

    // 获取template各自的主thread上有多少notify
    AlgResourceRequest level0TempRequest;
    CHK_RET(algTempLevel0.GetRes(level0TempRequest));
    notifyIdxControlToTemplates_.push_back(level0TempRequest.notifyNumOnMainThread);
#ifndef TEST_NO_LEVEL1
    AlgResourceRequest level1TempRequest;
    CHK_RET(algTempLevel1.GetRes(level1TempRequest));
    notifyIdxControlToTemplates_.push_back(level1TempRequest.notifyNumOnMainThread);
#endif
    notifyIdxTemplatesToControl_.push_back(0);
    notifyIdxTemplatesToControl_.push_back(1);
    HCCL_DEBUG("[%s]notifyIdxControlToTemplates_ size[%u]", __func__, notifyIdxControlToTemplates_.size());
    HCCL_DEBUG("[%s]notifyIdxTemplatesToControl_ size[%u]", __func__, notifyIdxTemplatesToControl_.size());

    // 单独本地拷贝使用 todo未定义变量
    // templateMainXThreads_.push_back(intraThreads_.at(0));
    // notifyIdxXControlToTemplates_.push_back(level0TempRequest.notifyNumOnMainThread);
    // notifyIdxXTemplatesToControl_.push_back(0);

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
HcclResult CcuV2ReduceScatterOmniPipeExecutor<AlgTopoMatch, InsAlgTempLevel0, InsAlgTempLevel1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[%s] Start", __func__);
    auto algHierarchyInfo = resCtx.algHierarchyInfo;
    // 重复的template构造
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
#ifndef TEST_NO_LEVEL1
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(2, 0));
    u32 index = 0;
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
#endif
    InsAlgTempLevel0 algTemplateLevel0(param, myRank_, subCommRanks0);
#ifndef TEST_NO_LEVEL1
    InsAlgTempLevel1 algTemplateLevel1(param, myRank_, subCommRanks1);
#endif

    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.inputSize = param.inputSize;
    tempAlgParamsCommon.buffInfo.outputSize = param.outputSize;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.inputSliceStride = dataSize_;
    tempAlgParamsCommon.outputSliceStride = dataSize_;
    tempAlgParamsCommon.localCopyFlag = 0;

    TemplateResource templateResourceCommon;
    templateResourceCommon.threads = resCtx.threads;
    if (param.engine == COMM_ENGINE_CCU) {
        HCCL_DEBUG("[%s] myRank[%u] param engine is CCU", __func__, myRank_);
    } else {
        HCCL_DEBUG("[%s] myRank[%u] param engine is not CCU", __func__, myRank_);
    }

    HCCL_DEBUG("[%s] remoteRankToChannelInfo size[%u]", __func__, remoteRankToChannelInfo_.size());
    TemplateResource templateResourceLevel0 = templateResourceCommon;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateResourceLevel0.channels = remoteRankToChannelInfo_[0];
        for (const auto& pair : templateResourceLevel0.channels) {
            auto key = pair.first;
            auto channelVec = pair.second;
            for (size_t i = 0; i < channelVec.size(); ++i) {
                HCCL_DEBUG("[%s]Level0 key[%u] (%u) remote[%u]", __func__, key, i, channelVec[i].remoteRank );
            }
        }
    }
#ifndef TEST_NO_LEVEL1
    TemplateResource templateResourceLevel1 = templateResourceCommon;
    if (remoteRankToChannelInfo_.size() > 1) {
        templateResourceLevel1.channels = remoteRankToChannelInfo_[1];
        for (const auto& pair : templateResourceLevel1.channels) {
            auto key = pair.first;
            auto channelVec = pair.second;
            for (size_t i = 0; i < channelVec.size(); ++i) {
                HCCL_DEBUG("[%s]Level1 key[%u] (%u) remote[%u]", __func__, key, i, channelVec[i].remoteRank );
            }
        }
    }
#endif

#ifndef TEST_NO_LEVEL1
    PrepareResForTemplate(param, resCtx, algTemplateLevel0, algTemplateLevel1);
#else
    PrepareResForTemplate(param, resCtx, algTemplateLevel0, algTemplateLevel0);
#endif
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
    // endpointAttrBwAvg = {1,1,1};
    endpointAttrBwAvg = {3,4,1};
#endif
    // 2、计算loop  搓一起调雪松的方法,返回的数组0是maxCountPerloop,1是loopTimes
#if 0
    OmniPipeScratchParam scratchParam;
    scratchParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    scratchParam.endpointAttrBw = endpointAttrBwAvg;
    // TODOv 测试值
    std::vector<u64> levelAlgType{1, 1, 1};
    // levelAlgType.push_back(algTemplateLevel0.CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT));
    // levelAlgType.push_back(algTemplateLevel1.CalcScratchMultiple(BufferType::DEFAULT, BufferType::DEFAULT));
    // levelAlgType.push_back(1); // 3D
    scratchParam.levelAlgType = levelAlgType;
    scratchParam.dataSize =  dataSize_;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.maxTmpMemSize = 200 * 1024 * 1024;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;

    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam);
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];
#endif
    u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%llu]", __func__, myRank_, loopTimes);

    // 3、计算n-1次loop的slice信息
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = maxCountPerLoop * dataTypeSize_;
    sliceParam.dataWholeSize = sliceParam.dataSizePerLoop;  // rs这个值和peerloop一致，已对齐
    sliceParam.endpointAttrBw = endpointAttrBwAvg;
    // sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_};
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    // sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    std::vector<u64> levelAlgType{1, 1, 1};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    OmniPipeSliceInfo alignSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        sliceParam.dataSizePerLoop = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        sliceParam.dataWholeSize = sliceParam.dataSizePerLoop;
        tailSliceInfo = CalcRSOmniPipeSliceInfo(sliceParam);
    }

    // 5、一次loop的数据处理
    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;
    TemplateDataParams tempAlgParamsLevel0 = tempAlgParamsCommon;
    TemplateDataParams tempAlgParamsLevel1 = tempAlgParamsCommon;
    // MARKv 全在UserIn做
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        // 5.2 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            omniPipeSliceInfo = tailSliceInfo;
        } else {
            omniPipeSliceInfo = alignSliceInfo;
        }

        auto innerServerStepNum = omniPipeSliceInfo.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] innerServerStepNum[%u]", __func__, myRank_, innerServerStepNum);

        if (omniPipeSliceInfo.isEmpty()) {
            HCCL_DEBUG("[%s] myRank[%u] omniPipeSliceInfo is Empty!", __func__, myRank_);
        } else {
            auto l0si = omniPipeSliceInfo.dataSliceLevel0;
            auto l1si = omniPipeSliceInfo.dataSliceLevel1;
            HCCL_DEBUG("[%s] myRank[%u] L0 stepNum[%u]", __func__, myRank_, l0si.size());
            HCCL_DEBUG("[%s] myRank[%u] L1 stepNum[%u]", __func__, myRank_, l1si.size());
        }

        // 5.4 for内层2d
        for (auto i = 0; i < innerServerStepNum; ++i) {
#ifndef TEST_NO_LEVEL1
            //第一步开始前同步
            CHK_RET(PreSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxControlToTemplates_));
#endif

            GenTemplateAlgParamsByDimData(tempAlgParamsLevel0, omniPipeSliceInfo.dataSliceLevel0[i],
                processedDataCount);
            templateResourceLevel0.ccuKernels.insert(templateResourceLevel0.ccuKernels.end(),
                resCtx.ccuKernels.begin(),
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);

            HCCL_DEBUG("[%s] myRank[%u] L0-step[%u] count[%u] inBuffBaseOff[%llu] outBuffBaseOff[%llu] inputSliceStride"
                       "[%llu] sliceSize[%llu]",
                __func__, myRank_, i, tempAlgParamsLevel0.count, tempAlgParamsLevel0.buffInfo.inBuffBaseOff,
                tempAlgParamsLevel0.buffInfo.outBuffBaseOff, tempAlgParamsLevel0.inputSliceStride,
                tempAlgParamsLevel0.sliceSize);

            for (auto j = 0; j < tempAlgParamsLevel0.inputOmniPipeSliceStride.size(); ++j) {
                HCCL_DEBUG("[%s] myRank[%u] L0-step[%u] inputOmniPipeSliceStride[%u]:[%llu]", __func__, myRank_,
                i, j, tempAlgParamsLevel0.inputOmniPipeSliceStride[j]);
            }

            CHK_RET(algTemplateLevel0.KernelRun(param, tempAlgParamsLevel0, templateResourceLevel0));
#ifndef TEST_NO_LEVEL1
            GenTemplateAlgParamsByDimData(tempAlgParamsLevel1, omniPipeSliceInfo.dataSliceLevel1[i],
                processedDataCount);
            templateResourceLevel1.ccuKernels.insert(templateResourceLevel1.ccuKernels.end(),
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);

            HCCL_DEBUG("[%s] myRank[%u] L1-step[%u] count[%u] inBuffBaseOff[%llu] outBuffBaseOff[%llu] inputSliceStride"
                       "[%llu] sliceSize[%llu]",
                __func__, myRank_, i, tempAlgParamsLevel1.count, tempAlgParamsLevel1.buffInfo.inBuffBaseOff,
                tempAlgParamsLevel1.buffInfo.outBuffBaseOff, tempAlgParamsLevel1.inputSliceStride,
                tempAlgParamsLevel1.sliceSize);

            for (auto j = 0; j < tempAlgParamsLevel1.inputOmniPipeSliceStride.size(); ++j) {
                HCCL_DEBUG("[%s] myRank[%u] L1-step[%u] inputOmniPipeSliceStride[%u]:[%llu]", __func__, myRank_,
                i, j, tempAlgParamsLevel1.inputOmniPipeSliceStride[j]);
            }

            CHK_RET(algTemplateLevel1.KernelRun(param, tempAlgParamsLevel1, templateResourceLevel1));

            //第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(controlThread_, templateMainThreads_, notifyIdxTemplatesToControl_));
#endif
        }
        processedDataCount += currDataCount;
    }
#if T_DESC("本地拷贝", false)
    tempAlgParamsLevel0.localCopyFlag = 1;
    HCCL_DEBUG("[%s] myRank[%u] start local copy", __func__, myRank_);
    CHK_RET(algTemplateLevel0.KernelRun(param, tempAlgParamsLevel0, templateResourceLevel0));
#endif

    HCCL_INFO("[%s]End.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
                                CcuV2ReduceScatterOmniPipe,
                                CcuV2ReduceScatterOmniPipeExecutor,
                                TopoMatchMultilevel,
                                CcuTempReduceScatterOmniPipeMesh1D,
                                CcuTempReduceScatterOmniPipeMesh1D);
}