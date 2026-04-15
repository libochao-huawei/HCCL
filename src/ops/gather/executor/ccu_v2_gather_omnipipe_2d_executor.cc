/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_v2_gather_omnipipe_2d_executor.h"
#include "ccu_temp_gather_omnipipe_mesh_1d.h"
#include "ccu_temp_gather_omnipipe_nhr_1d_mem2mem.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2GatherOmniPipe2DExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm,
    TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank; // 通信域 RankID
    rankSize_ = topoInfo->userRankSize; // 通信域的 Rank数量
    devType_ = topoInfo->deviceType; // 硬件类型
    // topomatch这边先写死
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    HCCL_INFO("MT, init algHierarchyInfo.infos");
    u32 userrank = topoInfo->userRank;
    HCCL_INFO("MT, userrank = %u, ranksize is %u", userrank, rankSize_);

// #if T_DESC("2x2用例", false) //TODO:change
#if T_DESC("2x2用例", true)
    if (userrank == 0 || userrank == 1) {
        algHierarchyInfo.infos = {{{0, 1}, {0, 1, 2, 3}}};
    } else {
        algHierarchyInfo.infos = {{{2, 3}, {0, 1, 2, 3}}};
    }
#endif

// #if T_DESC("4x2用例", true) //TODO:change
#if T_DESC("4x2用例", false)
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else {
        algHierarchyInfo.infos = {{{4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    }
#endif

    for (int i = 0; i < algHierarchyInfo.infos.size(); i++) {
        for (int j = 0; j < algHierarchyInfo.infos[i].size(); j++) {
            for (int k = 0; k < algHierarchyInfo.infos[i][j].size(); k++) {
                HCCL_DEBUG("rank id %u, i %d, j %d, k %d, %u", userrank, i, j, k, algHierarchyInfo.infos[i][j][k]);
            }
        }
    }

    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][CalcRes]  Start rank is %u", myRank_);

    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(2, 0));
    u64 count = 0;
    for (int i = myRank_ % algHierarchyInfo.infos[0][0].size(); i< algHierarchyInfo.infos[0][1].size(); i += algHierarchyInfo.infos[0][0].size()) {
        subCommRanks1[0][count++] = algHierarchyInfo.infos[0][1][i];
    }

    // 构建template
    InsAlgTemplate1 intraTempAlg(param, topoInfo->userRank, subCommRanks0);
    InsAlgTemplate0 interTempAlg(param, topoInfo->userRank, subCommRanks1);


    // 调用计算资源的函数
    AlgResourceRequest resReqIntra; // X 轴
    AlgResourceRequest resReqInter; // Y 轴

    intraTempAlg.CalcRes(comm, param, topoInfo, resReqIntra);
    interTempAlg.CalcRes(comm, param, topoInfo, resReqInter);

    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          resReqIntra.ccuKernelInfos.begin(),
                                          resReqIntra.ccuKernelInfos.end());

    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                          resReqInter.ccuKernelInfos.begin(),
                                          resReqInter.ccuKernelInfos.end());

    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                          resReqIntra.ccuKernelNum.begin(),
                                          resReqIntra.ccuKernelNum.end());

    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(),
                                          resReqInter.ccuKernelNum.begin(),
                                          resReqInter.ccuKernelNum.end());


    // 申请一条控制thread作为主thread，该thread仅用于两个template之间同步
    resourceRequest.notifyNumOnMainThread = 2;
    // 由于主thread被单独作为控制thread，因此总的slaveThread需要额外加上两个template的主thread
    resourceRequest.slaveThreadNum = resReqInter.slaveThreadNum + resReqIntra.slaveThreadNum + 2;
    // 第一个template的zhuthread需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqInter.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqInter.notifyNumPerThread.begin(),
                                              resReqInter.notifyNumPerThread.end());
    // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqIntra.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqIntra.notifyNumPerThread.begin(),
                                              resReqIntra.notifyNumPerThread.end());

    resourceRequest.channels.push_back(resReqIntra.channels[0]);
    resourceRequest.channels.push_back(resReqInter.channels[0]);

    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][CalcRes]  End");

    return HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsByDimData(
    const OpParam &param, TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount) {

    tempAlgParams.buffInfo.inBuffType = stepSliceInfo.buffInfo.inBuffType;
    tempAlgParams.buffInfo.outBuffType = stepSliceInfo.buffInfo.outBuffType;

    tempAlgParams.count = stepSliceInfo.count;  // 此斜对角step发送的数据量

    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;
    HCCL_DEBUG("myrank is %u, inBuffBaseOff is %llu, processedDataCount is %llu, end inBuffBaseOff is %llu",
        myRank_, stepSliceInfo.buffInfo.inBuffBaseOff, processedDataCount * dataTypeSize_, tempAlgParams.buffInfo.inBuffBaseOff);

    HCCL_DEBUG("myrank is %u, outBuffBaseOff is %llu, processedDataCount is %llu, end outBuffBaseOff is %llu",
        myRank_, stepSliceInfo.buffInfo.outBuffBaseOff, processedDataCount * dataTypeSize_, tempAlgParams.buffInfo.outBuffBaseOff);

    tempAlgParams.inputSliceStride = stepSliceInfo.inputSliceStride;
    tempAlgParams.outputSliceStride = stepSliceInfo.outputSliceStride;
    tempAlgParams.sliceSize = stepSliceInfo.sliceSize;

    // 新增
    tempAlgParams.inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride;
    tempAlgParams.outputOmniPipeSliceStride = stepSliceInfo.outputOmniPipeSliceStride;
    tempAlgParams.localCopyFlag = 0;

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitExectorInfo(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][InitExectorInfo]  Start");

    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[0][1].size() / algHierarchyInfo_.infos[0][0].size();

    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel1_;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;

    // 初始化root相关信息
    rootRank_ = param.root;  // 从参数中获取root节点
    rootIdxLevel0_ = rootRank_ % rankSizeLevel0_;
    rootIdxLevel1_ = rootRank_ / rankSizeLevel1_;

    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][InitExectorInfo] rootRank=%u, rootIdxLevel0=%u, rootIdxLevel1=%u",
              rootRank_, rootIdxLevel0_, rootIdxLevel1_);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::RestoreChannelMap(
    const AlgResourceCtxSerializable &resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>> &rankIdToChannelInfo)
{
    // todo:临时写法，待改进，两层边三层
    rankIdToChannelInfo.resize(3);
    for (u32 level = 0; level < resCtx.channels.size(); level++) {
        for (auto &channel: resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[level][remoteRank].push_back(channel);
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][Orchestrate] Orchestrate Start");

    // 参数填充
    // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    HCCL_DEBUG("thread size is %u", threads_.size());

    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(InitExectorInfo(param, resCtx));
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2GatherOmniPipe2DExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][Orchestrate] Orchestrate end");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &agAlgLevel0,
    InsAlgTemplate1 &agAlgLevel1)
{
    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor] [PrepareResForTemplate] begin");
    u64 l0ThreadsNum = agAlgLevel0.GetThreadNum();
    u64 l1ThreadsNum = agAlgLevel1.GetThreadNum();

    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + 1 + l0ThreadsNum);
    interThreads_.assign(threads_.begin() + 1 + l0ThreadsNum, threads_.end());

    HCCL_INFO("threads_ size %u, l0ThreadsNum %u, l1ThreadsNum %u", threads_.size(), l0ThreadsNum, l1ThreadsNum);

    // 用于两个算法同步
    controlThread_ = threads_.at(0);
    templateMainXYThreads_.push_back(intraThreads_.at(0));
    templateMainXYThreads_.push_back(interThreads_.at(0));

    // 获取两个template各自的主thread上有多少notify
    AlgResourceRequest resReqInter;
    AlgResourceRequest resReqIntra;
    CHK_RET(agAlgLevel0.GetRes(resReqInter));
    CHK_RET(agAlgLevel1.GetRes(resReqIntra));
    notifyIdxXYControlToTemplates_.push_back(resReqInter.notifyNumOnMainThread);
    notifyIdxXYControlToTemplates_.push_back(resReqIntra.notifyNumOnMainThread);
    notifyIdxXYTemplatesToControl_.push_back(0);
    notifyIdxXYTemplatesToControl_.push_back(1);

    HCCL_INFO("resReqInter.notifyNumOnMainThread %u, resReqIntra.notifyNumOnMainThread %u", resReqInter.notifyNumOnMainThread, resReqIntra.notifyNumOnMainThread);

    // 单独本地拷贝使用
    templateMainXThreads_.push_back(intraThreads_.at(0));
    notifyIdxXControlToTemplates_.push_back(resReqInter.notifyNumOnMainThread);
    notifyIdxXTemplatesToControl_.push_back(0);

    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor] [PrepareResForTemplate] end");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2GatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][OrchestrateLoop] Start");
    // 声明level0 templateargs
    TemplateDataParams tempAlgParamslevel0;
    tempAlgParamslevel0.inputSliceStride = dataSize_;
    tempAlgParamslevel0.outputSliceStride = dataSize_;// 未切分的整个数据块的大小
    tempAlgParamslevel0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamslevel0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamslevel0.buffInfo.inputSize = param.inputSize;
    tempAlgParamslevel0.buffInfo.outputSize = param.outputSize;

    // 声明level1 templateargs
    TemplateDataParams tempAlgParamslevel1;
    tempAlgParamslevel1.inputSliceStride = dataSize_;
    tempAlgParamslevel1.outputSliceStride = dataSize_;
    tempAlgParamslevel1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamslevel1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamslevel1.buffInfo.inputSize = param.inputSize;
    tempAlgParamslevel1.buffInfo.outputSize = param.outputSize;

    HCCL_DEBUG("kernel num size %u, kernel size %u", resCtx.ccuKernelNum.size(), resCtx.ccuKernels.size());

    HCCL_DEBUG("kernel num 0 %u, kernel num 1 %u, kernel size %u", resCtx.ccuKernelNum[0], resCtx.ccuKernelNum[1], resCtx.ccuKernels.size());



    // 构造level0 template资源
    TemplateResource templateResourcelevel0;
    templateResourcelevel0.channels = remoteRankToChannelInfo_[0];
    templateResourcelevel0.threads = resCtx.threads;
    templateResourcelevel0.ccuKernels.insert(templateResourcelevel0.ccuKernels.end(),
                                             resCtx.ccuKernels.begin(),
                                             resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);

    HCCL_DEBUG("x channel size is %u", templateResourcelevel0.channels.size());

    for (const auto& pair : templateResourcelevel0.channels) {
        u32 key = pair.first;
        const std::vector<ChannelInfo>& channelVec = pair.second;

        for (size_t i = 0; i < channelVec.size(); ++i) {
            HCCL_DEBUG("x key is %u, i is %u, remote is %u", key, i, channelVec[i].remoteRank );
        }
    }


    // 构造level1 template资源
    TemplateResource templateResourcelevel1;
    templateResourcelevel1.channels = remoteRankToChannelInfo_[1];
    templateResourcelevel1.threads = resCtx.threads;
    templateResourcelevel1.ccuKernels.insert(templateResourcelevel1.ccuKernels.end(),
                                             resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                                             resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);

    HCCL_DEBUG("y channel size is %u", templateResourcelevel1.channels.size());

    for (const auto& pair : templateResourcelevel1.channels) {
        u32 key = pair.first;
        const std::vector<ChannelInfo>& channelVec = pair.second;

        for (size_t i = 0; i < channelVec.size(); ++i) {
            HCCL_DEBUG("y key is %u, i is %u, remote is %u", key, i, channelVec[i].remoteRank );
        }
    }

    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo_.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(2, 0));

    u64 count = 0;
    for (int i = myRank_ % algHierarchyInfo_.infos[0][0].size(); i< algHierarchyInfo_.infos[0][1].size(); i += algHierarchyInfo_.infos[0][0].size()) {
        subCommRanks1[0][count++] = algHierarchyInfo_.infos[0][1][i];
    }

    // 构建level0 template
    std::shared_ptr<InsAlgTemplate0> algTemplatelevel0 = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    // 构建level1 template
    std::shared_ptr<InsAlgTemplate1> algTemplatelevel1 = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);

    for (int i = 0; i < subCommRanks0.size(); i++) {
        for (int j = 0; j < subCommRanks0[0].size(); j++) {
            HCCL_DEBUG("2 rankid is %u, subCommRanks0 [%d][%d] is %u", myRank_, i, j, subCommRanks0[i][j]);
        }
    }

    for (int i = 0; i < subCommRanks1.size(); i++) {
        for (int j = 0; j < subCommRanks1[0].size(); j++) {
            HCCL_DEBUG("2 rankid is %u, subCommRanks1 [%d][%d] is %u", myRank_, i, j, subCommRanks1[i][j]);
        }
    }

    PrepareResForTemplate(param, resCtx, *algTemplatelevel0, *algTemplatelevel1);

    HCCL_DEBUG("MT mock the endpointAttrBwNew");
    std::vector<EndpointAttrBwCoeff> endpointAttrBwNew = {1,1,1};

    // 2、计算loop  ccu 不用cclbuff，根据UB_MAX_DATA_SIZE来计算
    u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);

    OmniPipeSliceParam omniPipeSliceParam;
    omniPipeSliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    omniPipeSliceParam.endpointAttrBw = endpointAttrBwNew;
    omniPipeSliceParam.dataSizePerLoop = maxCountPerLoop * dataTypeSize_;
    omniPipeSliceParam.dataTypeSize = dataTypeSize_;
    omniPipeSliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    omniPipeSliceParam.opMode = param.opMode;
    omniPipeSliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    omniPipeSliceParam.dataWholeSize = dataCount_ * dataTypeSize_;
    std::vector<u64> levelAlgType{1, 1, 1}; // 后面再修改
    omniPipeSliceParam.levelAlgType = levelAlgType;

    OmniPipeSliceInfo alignSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        omniPipeSliceParam.dataSizePerLoop = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        omniPipeSliceParam.dataWholeSize = dataCount_ * dataTypeSize_;
        tailSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
    }

    // 5、进行一次loop的数据处理
    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;
    for (u64 loop = 0; loop < loopTimes; loop++) {

        // 5.2 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfo.isEmpty()) {
            omniPipeSliceInfo = tailSliceInfo;
            HCCL_INFO("use tail sliceinfo");
        } else {
            omniPipeSliceInfo = alignSliceInfo;
            HCCL_INFO("use align sliceinfo");
        }


        CHK_RET(PreSyncInterThreads(controlThread_, templateMainXThreads_, notifyIdxXControlToTemplates_));
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // 本地拷贝 - 只有root节点需要接收数据，其他节点只需要发送
        if (myRank_ == rootRank_) {
            tempAlgParamslevel0.buffInfo.inBuffType = BufferType::INPUT;
            tempAlgParamslevel0.count = currDataCount;
            tempAlgParamslevel0.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
            tempAlgParamslevel0.inputSliceStride = dataCount_ * dataTypeSize_;
            tempAlgParamslevel0.outputSliceStride = dataCount_ * dataTypeSize_;
            tempAlgParamslevel0.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
            tempAlgParamslevel0.repeatNum = rankSize_;
            tempAlgParamslevel0.sliceSize = currDataCount * dataTypeSize_;
            tempAlgParamslevel0.localCopyFlag = 1;
            tempAlgParamslevel0.inputOmniPipeSliceStride.push_back(0);
            tempAlgParamslevel0.outputOmniPipeSliceStride.push_back(0);

            HCCL_DEBUG("dataCount_ %llu, processedDataCount %llu, maxCountPerLoop %llu", dataCount_, processedDataCount, maxCountPerLoop);
            HCCL_DEBUG("sliceSize %llu, currDataCount %llu, dataTypeSize_ %llu", tempAlgParamslevel0.sliceSize, currDataCount, dataTypeSize_);

            CHK_RET(algTemplatelevel0->KernelRun(param, tempAlgParamslevel0, templateResourcelevel0));
            CHK_RET(PostSyncInterThreads(controlThread_, templateMainXThreads_, notifyIdxXTemplatesToControl_));
            HCCL_DEBUG("local copy end");
        }

        u32 level0StepCount = omniPipeSliceInfo.dataSliceLevel0.size();
        HCCL_DEBUG("level0StepCount %u", level0StepCount);

        // 5.3 for内层2d
        for (u32 i = 0; i < level0StepCount; i++) {
            // Gather关键差异：判断当前节点是否需要参与此次数据传输
            // 只有与root同行或同列的节点才需要转发数据，其他节点只发送自己的数据
            
            // 判断当前步骤是否涉及与root同行或同列的通信
            bool needExecute = false;
            
            // 如果当前节点就是root，必须执行
            if (myRank_ == rootRank_) {
                needExecute = true;
            }
            // 如果当前节点与root在同一行（level1相同），需要执行level0的通信
            else if (rankIdxLevel1_ == rootIdxLevel1_) {
                needExecute = true;
            }
            // 如果当前节点与root在同一列（level0相同），需要执行level1的通信
            else if (rankIdxLevel0_ == rootIdxLevel0_) {
                needExecute = true;
            }
            
            // 只有需要执行的节点才参与此次通信
            if (!needExecute) {
                HCCL_DEBUG("rank %u skip step %u, not in root's row or column", myRank_, i);
                continue;
            }

            // 初始化机内template param
            GenTemplateAlgParamsByDimData(param, tempAlgParamslevel0, omniPipeSliceInfo.dataSliceLevel0[i], processedDataCount);
            GenTemplateAlgParamsByDimData(param, tempAlgParamslevel1, omniPipeSliceInfo.dataSliceLevel1[i], processedDataCount);

            HCCL_DEBUG("x loop %llu, rankid %u, count %u, inBuffBaseOff %llu, outBuffBaseOff %llu, inputSliceStride %llu, outputSliceStride %llu, sliceSize %llu",
                      i, myRank_, tempAlgParamslevel0.count, tempAlgParamslevel0.buffInfo.inBuffBaseOff, tempAlgParamslevel0.buffInfo.outBuffBaseOff,
                      tempAlgParamslevel0.inputSliceStride, tempAlgParamslevel0.outputSliceStride, tempAlgParamslevel0.sliceSize);

            for (int i = 0; i < tempAlgParamslevel0.inputOmniPipeSliceStride.size(); i++) {
                HCCL_DEBUG("x input rankid %u, diag %d is %llu", myRank_, i, tempAlgParamslevel0.inputOmniPipeSliceStride[i]);
                HCCL_DEBUG("x output rankid %u, diag %d is %llu", myRank_, i, tempAlgParamslevel0.outputOmniPipeSliceStride[i]);
            }

            HCCL_DEBUG("y loop %llu, rankid %u, count %u, inBuffBaseOff %llu, outBuffBaseOff %llu, inputSliceStride %llu, outputSliceStride %llu, sliceSize %llu",
                      i, myRank_, tempAlgParamslevel1.count, tempAlgParamslevel1.buffInfo.inBuffBaseOff, tempAlgParamslevel1.buffInfo.outBuffBaseOff,
                      tempAlgParamslevel1.inputSliceStride, tempAlgParamslevel1.outputSliceStride, tempAlgParamslevel1.sliceSize);

            for (int i = 0; i < tempAlgParamslevel1.inputOmniPipeSliceStride.size(); i++) {
                HCCL_DEBUG("y input rankid %u, diag %d is %llu", myRank_, i, tempAlgParamslevel1.inputOmniPipeSliceStride[i]);
                HCCL_DEBUG("y output rankid %u, diag %d is %llu", myRank_, i, tempAlgParamslevel1.outputOmniPipeSliceStride[i]);
            }

            //第一步开始前同步
            CHK_RET(PreSyncInterThreads(controlThread_, templateMainXYThreads_, notifyIdxXYControlToTemplates_));

            // 执行机内template任务
            CHK_RET(algTemplatelevel0->KernelRun(param, tempAlgParamslevel0, templateResourcelevel0));
            CHK_RET(algTemplatelevel1->KernelRun(param, tempAlgParamslevel1, templateResourcelevel1));

            //第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(controlThread_, templateMainXYThreads_, notifyIdxXYTemplatesToControl_));
        }
        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2GatherOmniPipe2DExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;

}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_GATHER,
                                CcuGatherOmniPipe2D,
                                InsV2GatherOmniPipe2DExecutor,
                                TopoMatchUBX,
                                CcuTempGatherOmniPipeMesh1D,
                                CcuTempGatherOmniPipeMesh1D);

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_GATHER,
                              CcuGatherOmniPipe2DNHR,
                              InsV2GatherOmniPipe2DExecutor,
                              TopoMatchUBX,
                              CcuTempGatherOmniPipeNHR1DMem2Mem,
                              CcuTempGatherOmniPipeNHR1DMem2Mem);

}
