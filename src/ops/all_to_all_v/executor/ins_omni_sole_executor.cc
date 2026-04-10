/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "template_utils.h"
#include "ins_omni_sole_executor.h"


namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsOmniSoleExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_INFO("[InitCommInfo] begin ");
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];

    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;

    // dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    const u64* data = reinterpret_cast<const u64*>(param.varData);
    dataCount_ = data[0];
    dataSize_ = dataCount_ * dataTypeSize_;
    HCCL_INFO("[InsOmniSoleExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], dataType_ [%u], "
        "dataCount_ [%llu]", myRank_, rankSize_, devType_, dataType_, dataCount_);

    HCCL_INFO("[InsOmniSoleExecutor][InitCommInfo] dataTypeSize_ [%u], dataSize_ [%llu]", dataTypeSize_, dataSize_);
    return HCCL_SUCCESS;
}

uint32_t InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ReadBits(std::ifstream& file, uint64_t offset, size_t numBits) {
    uint32_t result = 0;
    size_t bitsRead = 0;
    const uint64_t byteOffset = offset / 8;
    const size_t bitOffset = offset % 8;
    const size_t totalBits = bitOffset + numBits;
    const size_t totalBytes = (totalBits + 7) / 8;
    std::vector<uint8_t> bytes(totalBytes, 0);

    file.clear();
    file.seekg(byteOffset, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), totalBytes)) {
        return 0;
    }
    while (bitsRead < numBits) {
        const size_t absBit = bitOffset + bitsRead;
        const size_t curByteIdx = absBit / 8;
        const size_t curBitIdx = absBit % 8;
        const uint8_t bit = (bytes[curByteIdx] >> (7 - curBitIdx)) & 0x1;
        result = (result << 1) | bit;
        bitsRead++;
    }
    return result;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ParseXmlInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    (void)param;
    (void)topoInfo;
    const char *omniBinPath = std::getenv("HCCL_OMNI_BIN_PATH");
    std::vector<std::string> candidatePaths;
    if (omniBinPath != nullptr && omniBinPath[0] != '\0') {
        candidatePaths.emplace_back(omniBinPath);
        candidatePaths.emplace_back(std::string(omniBinPath) + "\\rank_" + std::to_string(myRank_) + ".bin");
    }
    candidatePaths.emplace_back("rank_" + std::to_string(myRank_) + ".bin");
    candidatePaths.emplace_back("example.bin");

    std::ifstream file;
    std::string selectedPath;
    for (const auto &candidatePath : candidatePaths) {
        file.open(candidatePath, std::ios::binary);
        if (file.is_open()) {
            selectedPath = candidatePath;
            break;
        }
        file.clear();
    }
    if (!file) {
        HCCL_ERROR("[InsOmniSoleExecutor][ParseXmlInfo] can not open omni bin file, env[%s], rank[%u].",
            omniBinPath == nullptr ? "" : omniBinPath, myRank_);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[InsOmniSoleExecutor][ParseXmlInfo] use omni bin file [%s].", selectedPath.c_str());

    XmlInfo xmlInfo;
    uint64_t offset = 0;
    uint16_t op = ReadBits(file, offset, 8);
    offset += 8;
    uint16_t notifyNumOnMainThread = ReadBits(file, offset, 8);
    offset += 8;
    uint16_t notifyNumPerThread = ReadBits(file, offset, 8);
    offset += 8;
    uint16_t netLayer = ReadBits(file, offset, 8);
    offset += 8;
    uint16_t chanCount = ReadBits(file, offset, 8);
    offset += 8;
    (void)op;
    xmlInfo.resInfo.notifyNumOnMainThread = notifyNumOnMainThread;
    xmlInfo.resInfo.notifyNumPerThread = notifyNumPerThread;
    xmlInfo.resInfo.netLayerNum = netLayer;

    std::map<u32, OmniChannelInfo> mapChannelInfo;
    for (uint16_t i = 0; i < chanCount; i++) {
        OmniChannelInfo omniChannelInfo;
        uint16_t channelId = ReadBits(file, offset, 8);
        offset += 8;
        uint16_t localRank = ReadBits(file, offset, 8);
        offset += 8;
        uint16_t remoteRank = ReadBits(file, offset, 8);
        offset += 8;
        uint16_t linkProto = ReadBits(file, offset, 8);
        offset += 8;
        if (localRank != myRank_) {
            continue;
        }

        omniChannelInfo.channelId = channelId;
        omniChannelInfo.remoteRank = remoteRank;
        omniChannelInfo.channelProtocol = static_cast<CommProtocol>(linkProto);
        mapChannelInfo[remoteRank] = omniChannelInfo;
    }
    xmlInfo.resInfo.mapchannelInfo.push_back(mapChannelInfo);

    do {
        OmniSendRecvInfo omniSendRecvInfo;
        uint16_t opcode = ReadBits(file, offset, 5);
        offset += 5;
        uint16_t linkType = ReadBits(file, offset, 2);
        offset += 2;
        uint16_t linkProto = ReadBits(file, offset, 3);
        offset += 3;
        uint16_t localRankID = ReadBits(file, offset, 10);
        offset += 10;
        uint16_t sliceNum = ReadBits(file, offset, 10);
        offset += 10;
        uint16_t srcSliceCnt = ReadBits(file, offset, 4);
        offset += 4;
        uint16_t dstSliceCnt = ReadBits(file, offset, 4);
        offset += 4;
        uint16_t notifyFlag = ReadBits(file, offset, 1);
        offset += 1;
        uint16_t notifyThread = ReadBits(file, offset, 4);
        offset += 4;
        uint16_t waitFlag = ReadBits(file, offset, 1);
        offset += 1;
        uint16_t waitThread = ReadBits(file, offset, 4);
        offset += 4;
        offset += 16;

        uint16_t sendChannelID = ReadBits(file, offset, 8);
        offset += 8;
        uint16_t recvChannelID = ReadBits(file, offset, 8);
        offset += 8;
        uint16_t threadIdx = ReadBits(file, offset, 5);
        offset += 5;
        uint16_t instructionID = ReadBits(file, offset, 16);
        offset += 16;
        uint16_t reduceType = ReadBits(file, offset, 2);
        offset += 2;
        uint16_t inputDataType = ReadBits(file, offset, 4);
        offset += 4;
        uint16_t outputDataType = ReadBits(file, offset, 4);
        offset += 4;
        offset += 17;
        (void)linkProto;
        (void)localRankID;
        (void)notifyFlag;
        (void)notifyThread;
        (void)waitFlag;
        (void)waitThread;
        (void)sendChannelID;
        (void)recvChannelID;
        (void)instructionID;

        omniSendRecvInfo.optype = static_cast<OpType>(opcode);
        omniSendRecvInfo.inputDataType = static_cast<HcclDataType>(inputDataType);
        omniSendRecvInfo.outputDataType = static_cast<HcclDataType>(outputDataType);
        omniSendRecvInfo.reduceType = static_cast<HcclReduceOp>(reduceType);
        omniSendRecvInfo.sliceNum = sliceNum;
        omniSendRecvInfo.linkType = linkType;
        omniSendRecvInfo.threadIdx = threadIdx;

        for (uint16_t i = 0; i < srcSliceCnt; i++) {
            OmniSliceInfo omniSliceInfo;
            uint16_t srcBufferType = ReadBits(file, offset, 2);
            offset += 2;
            uint16_t srcSliceIdx = ReadBits(file, offset, 10);
            offset += 10;
            uint16_t remoteRank = ReadBits(file, offset, 10);
            offset += 10;
            omniSliceInfo.sliceType = srcBufferType;
            omniSliceInfo.sliceIdx = srcSliceIdx;
            omniSliceInfo.remoteRank = remoteRank;
            omniSendRecvInfo.srcSliceInfo.push_back(omniSliceInfo);
        }

        for (uint16_t i = 0; i < dstSliceCnt; i++) {
            OmniSliceInfo omniSliceInfo;
            uint16_t dstBufferType = ReadBits(file, offset, 2);
            offset += 2;
            uint16_t dstSliceIdx = ReadBits(file, offset, 10);
            offset += 10;
            uint16_t remoteRank = ReadBits(file, offset, 10);
            offset += 10;

            omniSliceInfo.sliceType = dstBufferType;
            omniSliceInfo.sliceIdx = dstSliceIdx;
            omniSliceInfo.remoteRank = remoteRank;
            omniSendRecvInfo.dstSliceInfo.push_back(omniSliceInfo);
        }
        xmlInfo.vecSendRecvInfo.push_back(omniSendRecvInfo);
    } while(file.peek() != EOF);

    xmlInfo_ = std::move(xmlInfo);
    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo));
    CHK_RET(ParseXmlInfo(param, topoInfo)); // 解析xml

    std::vector<std::vector<u32>> tempAlgHierachyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        tempAlgHierachyInfo.push_back(algHierarchyInfo.infos[0][1]);    // clos拓扑，包含所有rank
    } else {
        tempAlgHierachyInfo = algHierarchyInfo.infos[0];
    }

    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate = 
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierachyInfo);
    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest, xmlInfo_);

    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsOmniSoleExecutor][Orchestrate] Orchestrate Start, rankid [%u]", myRank_);

    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo));

    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsOmniSoleExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("[InsOmniSoleExecutor][Orchestrate] Orchestrate End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsOmniSoleExecutor][OrchestrateLoop] Start, rankid [%u]", myRank_);
    
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    // 准备资源
    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    } else if (param.engine == COMM_ENGINE_AIV) {
        templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    }
    templateAlgRes.threads = resCtx.threads;

    u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    if (param.engine == COMM_ENGINE_AIV) {
        maxCountPerLoop = std::max<u64>(1, resCtx.cclMem.size / dataTypeSize_);
    }
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_INFO("[InsOmniSoleExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);

    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        const u64 sliceNum = xmlInfo_.vecSendRecvInfo.empty() ? 1 : std::max<u64>(1, xmlInfo_.vecSendRecvInfo[0].sliceNum);
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_ / sliceNum;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams.repeatNum = 1;  // 不需要重复
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;
        tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[InsOmniSoleExecutor][OrchestrateLoop] End, rankid [%u]", myRank_);
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV,
                CcuOMNI,
                InsOmniSoleExecutor,
                TopoMatch1D,
                CcuTempOmni);

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV,
                AivHcclOmni,
                InsOmniSoleExecutor,
                TopoMatch1D,
                AivTempOmni);

}
