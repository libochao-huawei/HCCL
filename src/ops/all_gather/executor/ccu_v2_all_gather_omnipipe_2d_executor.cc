/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "ccu_v2_all_gather_omnipipe_2d_executor.h"
#include "ccu_temp_all_gather_omnipipe_mesh_1d.h"
// #include "ccu_temp_all_gather_omnipipe_nhr_1d_mem2mem.h"
 
namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllGatherOmniPipe2DExecutor()
{
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm,
    TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
 
#if T_DESC("正常拓扑计算", false)
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
#endif
 
    u32 userrank = topoInfo->userRank;
#if T_DESC("2x2用例", true)
    if (userrank == 0 || userrank == 1) {
        algHierarchyInfo.infos = {{{0, 1}, {0, 1, 2, 3}}};
    } else {
        algHierarchyInfo.infos = {{{2, 3}, {0, 1, 2, 3}}};
    }
#endif
 
#if T_DESC("4x2用例", false)
    HCCL_DEBUG("[%s] 4x2-TestCase", __func__);
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else {
        algHierarchyInfo.infos = {{{4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    }
#endif
 
#if T_DESC("2x4用例", false)
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
#endif
 
#if T_DESC("4x4用例", false)
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    } else if (userrank == 4 || userrank == 5 || userrank == 6 || userrank == 7) {
        algHierarchyInfo.infos = {{{4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    } else if (userrank == 8 || userrank == 9 || userrank == 10 || userrank == 11) {
        algHierarchyInfo.infos = {{{8, 9,  10, 11}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    } else if (userrank == 12 || userrank == 13 || userrank == 14 || userrank == 15) {
        algHierarchyInfo.infos = {{{12, 13, 14, 15}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    }
#endif
 
    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_INFO("[%s] myRank[%u] (%d, %d, %d) %u", __func__, topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }
 
    return HCCL_SUCCESS;
}
 
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[%s] myRank[%u] start", __func__, myRank_);
    HCCL_DEBUG("[%s] myRank[%u] algHierarchyInfo.infos[0][1]size[%u] algHierarchyInfo.infos[0][0]size[%u]", __func__,
        myRank_, algHierarchyInfo.infos[0][1].size(), algHierarchyInfo.infos[0][0].size());
 
    // 重复的template构造
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    auto size = algHierarchyInfo.infos[0][1].size() / algHierarchyInfo.infos[0][0].size();
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
    auto i = 0;
    for (auto board: subCommRanks0) {
        for (auto rank : board) {
            HCCL_DEBUG("[%s] myRank[%u] subCommLevel0[%d] rank[%u]", __func__, myRank_, i, rank);
        }
        ++i;
    }
 
    u32 index = 0;
    for (int i = myRank_ % algHierarchyInfo.infos[0][0].size(); i < algHierarchyInfo.infos[0][1].size();
         i += algHierarchyInfo.infos[0][0].size()) {
        subCommRanks1[0][index++] = algHierarchyInfo.infos[0][1][i];
    }
 
    i = 0;
    for (auto board: subCommRanks1) {
        for (auto rank : board) {
            HCCL_DEBUG("[%s] myRank[%u] subCommLevel1[%d] rank[%u]", __func__, myRank_, i, rank);
        }
        ++i;
    }
    // 构建template
    InsAlgTemplate1 intraTempAlg(param, topoInfo->userRank, subCommRanks0);
    InsAlgTemplate0 interTempAlg(param, topoInfo->userRank, subCommRanks1);
 
 
    // 调用计算资源的函数
    AlgResourceRequest resReqLevel0; // X 轴
    AlgResourceRequest resReqLevel1; // Y 轴
 
    intraTempAlg.CalcRes(comm, param, topoInfo, resReqLevel0);
    interTempAlg.CalcRes(comm, param, topoInfo, resReqLevel1);
 
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
    resourceRequest.slaveThreadNum = resReqLevel1.slaveThreadNum + resReqLevel0.slaveThreadNum + 2;
    // 第一个template的zhuthread需要的notify数量，+1是因为需要和控制thread做同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel1.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqLevel1.notifyNumPerThread.begin(),
                                              resReqLevel1.notifyNumPerThread.end());
    // 这一条是interTemplate的主thread，需要+1是为了和控制thread进行同步
    resourceRequest.notifyNumPerThread.emplace_back(resReqLevel0.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqLevel0.notifyNumPerThread.begin(),
                                              resReqLevel0.notifyNumPerThread.end());
 
 
    HCCL_DEBUG("[%s] myRank[%u] resReqLevel0 channels size[%u]", __func__, myRank_, resReqLevel0.channels.size());
    HCCL_DEBUG("[%s] myRank[%u] resReqLevel1 channels size[%u]", __func__, myRank_, resReqLevel1.channels.size());
    resourceRequest.channels.push_back(resReqLevel0.channels[0]);
    resourceRequest.channels.push_back(resReqLevel1.channels[0]);
 
    HCCL_DEBUG("[%s] myRank[%u] end", __func__, myRank_);
    return HCCL_SUCCESS;
}
 
// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsByDimData(
    const OpParam &param, TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount) {
 
    tempAlgParams.buffInfo.inBuffType = stepSliceInfo.buffInfo.inBuffType;
    tempAlgParams.buffInfo.outBuffType = stepSliceInfo.buffInfo.outBuffType;
 
    tempAlgParams.count = 0;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;
 
    HCCL_DEBUG("myrank is %u, inBuffBaseOff is %llu, processedDataCount is %llu, end inBuffBaseOff is %llu", myRank_,
        stepSliceInfo.buffInfo.inBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.inBuffBaseOff);
 
    HCCL_DEBUG("myrank is %u, outBuffBaseOff is %llu, processedDataCount is %llu, end outBuffBaseOff is %llu", myRank_,
        stepSliceInfo.buffInfo.outBuffBaseOff, processedDataCount, tempAlgParams.buffInfo.outBuffBaseOff);
 
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    HCCL_DEBUG("[%s] inputSliceStrie.size[%u]", __func__, tempAlgParams.stepSliceInfo.stepInputSliceStride.size());
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
 
    tempAlgParams.localCopyFlag = 0;
 
    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitExectorInfo(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherOmniPipe2DExecutor][InitExectorInfo]  Start");
 
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
 
    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[0][1].size() / algHierarchyInfo_.infos[0][0].size();
 
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    // rankIdxLevel1_ = myRank_ / rankSizeLevel1_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    HCCL_DEBUG("[%s] myRank[%u] X(%u)  Y(%u)", __func__, myRank_, rankIdxLevel0_, rankIdxLevel1_);
 
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
 
    return HcclResult::HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::RestoreChannelMap(
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
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherOmniPipe2DExecutor][Orchestrate] Orchestrate Start");
 
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
        HCCL_ERROR("[InsV2AllGatherOmniPipe2DExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
 
    HCCL_INFO("[InsV2AllGatherOmniPipe2DExecutor][Orchestrate] Orchestrate end");
 
    return HcclResult::HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &agAlgLevel0,
    InsAlgTemplate1 &agAlgLevel1)
{
    HCCL_INFO("[InsV2AllGatherOmniPipe2DExecutor] [PrepareResForTemplate] begin");
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
 
    HCCL_INFO("[InsV2AllGatherOmniPipe2DExecutor] [PrepareResForTemplate] end");
    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherOmniPipe2DExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherOmniPipe2DExecutor][OrchestrateLoop] Start");
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
    auto size = algHierarchyInfo_.infos[0][1].size() /algHierarchyInfo_.infos[0][0].size();
    std::vector<std::vector<u32>> subCommRanks1(1, std::vector<u32>(size, 0));
 
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
 
    // 1、计算带宽，平均带宽还是总带宽，如果是总带宽这边要处理成平均带宽
    // std::vector<std::vector<EndpointAttrBwCoeff>> endpointAttrBw;
    // CHK_RET(CalAllLevelEndpointAttrBwCoeff(param.hcclComm, myRank_, 3, endpointAttrBw));
 
    // 需要转化成平均带宽
    // std::vector<EndpointAttrBwCoeff> endpointAttrBwNew;
    // u64 bwIndex = 0;
    // for (u64 i = 0; i < endpointAttrBw.size(); i++) {
    //     for (u64 j = 0; j < endpointAttrBw[i].size(); ++j) {
    //         endpointAttrBw[i][j] /= algHierarchyInfo_.infos[i][j].size() - 1;
    //         endpointAttrBwNew[bwIndex++] = endpointAttrBw[i][j];
    //     }
    // }
 
    // 2、计算loop  ccu 不用cclbuff，根据UB_MAX_DATA_SIZE来计算
#if T_DESC("手动切Loop测试", true)
    // 131072 对应 4P datasize:1M
    // u64 maxCountPerLoop = static_cast<u64>(131072) / dataTypeSize_;
    u64 maxCountPerLoop = static_cast<u64>(256) / dataTypeSize_;
#else
    u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
#endif
    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    HCCL_DEBUG("[%s] dataSize_[%u] rankSize_[%u]", __func__, dataSize_, rankSize_);
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s] perLoopSize[%llu]", __func__, perLoopSize);
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, dataSize_);
 
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_INFO("[%s] loopTimes = [%u]", __func__, loopTimes);
 
    OmniPipeSliceParam omniPipeSliceParam;
    omniPipeSliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    omniPipeSliceParam.endpointAttrBw = {1, 1, 1};
    omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
    omniPipeSliceParam.dataTypeSize = dataTypeSize_;
    omniPipeSliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    omniPipeSliceParam.opMode = param.opMode;
    omniPipeSliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    omniPipeSliceParam.dataWholeSize = dataWholeSize;
    std::vector<u64> levelAlgType{1, 1, 1}; // 后面再修改
    omniPipeSliceParam.levelAlgType = levelAlgType;
 
    OmniPipeSliceInfo alignSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
    // TODOv remove debug
    for (auto item : alignSliceInfo.dataSliceLevel0) {
            for (auto i : item.stepInputSliceStride) {
                HCCL_DEBUG("[%s] XXXXX myRank[%u] L0 stepInputSliceStride=%u", __func__, myRank_, i);
            }
    }
 
    for (auto item : alignSliceInfo.dataSliceLevel1) {
            for (auto i : item.stepInputSliceStride) {
                HCCL_DEBUG("[%s] XXXXX myRank[%u] L1 stepInputSliceStride=%u", __func__, myRank_, i);
            }
    }
 
    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, dataSize_);
        omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
        omniPipeSliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
    }
 
    // 5、进行一次loop的数据处理
    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;
    for (u64 loop = 0; loop < loopTimes; loop++) {
 
        // 5.2 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && dataCount_ % maxCountPerLoop != 0) {
            omniPipeSliceInfo = tailSliceInfo;
            HCCL_INFO("[%s] loop[%lu] use tail sliceinfo", __func__, loop);
        } else {
            omniPipeSliceInfo = alignSliceInfo;
            HCCL_INFO("[%s] loop[%lu] use align sliceinfo", __func__, loop);
        }
 
 
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
#if 1
        // 本地拷贝
        CHK_RET(PreSyncInterThreads(controlThread_, templateMainXThreads_, notifyIdxXControlToTemplates_));
        tempAlgParamslevel0.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamslevel0.count = currDataCount;
        tempAlgParamslevel0.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamslevel0.inputSliceStride = dataCount_ * dataTypeSize_;
        tempAlgParamslevel0.outputSliceStride = dataCount_ * dataTypeSize_;
        tempAlgParamslevel0.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamslevel0.repeatNum = rankSize_;
        tempAlgParamslevel0.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamslevel0.localCopyFlag = 1;
 
        HCCL_DEBUG("[%s][LocalCopyInExecutor] dataCount_ %llu, processedDataCount %llu, maxCountPerLoop %llu", __func__, dataCount_,
            processedDataCount, maxCountPerLoop);
        HCCL_DEBUG("sliceSize %llu, currDataCount %llu, dataTypeSize_ %llu", tempAlgParamslevel0.sliceSize,
            currDataCount, dataTypeSize_);
 
        CHK_RET(algTemplatelevel0->KernelRun(param, tempAlgParamslevel0, templateResourcelevel0));
        CHK_RET(PostSyncInterThreads(controlThread_, templateMainXThreads_, notifyIdxXTemplatesToControl_));
        HCCL_DEBUG("local copy end");
        
#endif
 
        u32 level0StepCount = omniPipeSliceInfo.dataSliceLevel0.size();
        HCCL_DEBUG("level0StepCount %u", level0StepCount);
        // 5.3 for内层2d
        for (u32 i = 0; i < level0StepCount; i++) {
            // 初始化机内template param
            GenTemplateAlgParamsByDimData(param, tempAlgParamslevel0, omniPipeSliceInfo.dataSliceLevel0[i], processedDataCount);
            GenTemplateAlgParamsByDimData(param, tempAlgParamslevel1, omniPipeSliceInfo.dataSliceLevel1[i], processedDataCount);
 
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
    HCCL_INFO("[%s] End.", __func__);
    return HCCL_SUCCESS;
 
}
 
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER,
                                CcuAllGatherOmniPipe2D,
                                InsV2AllGatherOmniPipe2DExecutor,
                                TopoMatchUBX,
                                CcuTempAllGatherOmniPipeMesh1D,
                                CcuTempAllGatherOmniPipeMesh1D);
 
#if 0
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER,
                              CcuAllGatherOmniPipe2DNHR,
                              InsV2AllGatherOmniPipe2DExecutor,
                              TopoMatchUBX,
                              CcuTempAllGatherOmniPipeNHR1DMem2Mem,
                              CcuTempAllGatherOmniPipeNHR1DMem2Mem);
#endif
 
}