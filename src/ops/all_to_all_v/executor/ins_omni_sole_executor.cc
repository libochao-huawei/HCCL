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
constexpr uint16_t OMNI_RES_REQUEST_OPCODE = 0;
constexpr uint16_t OMNI_PRESYNC_OPCODE = 1;
constexpr uint16_t OMNI_POSTSYNC_OPCODE = 2;

std::string JoinOmniPath(const std::string &basePath, const std::string &fileName)
{
    if (basePath.empty()) {
        return fileName;
    }
    const char lastChar = basePath.back();
    if (lastChar == '\\' || lastChar == '/') {
        return basePath + fileName;
    }
    const char separator = (basePath.find('\\') != std::string::npos) ? '\\' : '/';
    return basePath + separator + fileName;
}

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

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ParseXmlInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    (void)topoInfo;
    const char *omniBinPath = std::getenv("HCCL_OMNI_BIN_PATH");
    std::vector<std::string> candidatePaths;
    if (omniBinPath != nullptr && omniBinPath[0] != '\0') {
        candidatePaths.emplace_back(omniBinPath);
        candidatePaths.emplace_back(JoinOmniPath(omniBinPath, "rank_" + std::to_string(myRank_) + ".bin"));
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

    uint64_t offset = 0;
    do {
        file.seekg(offset);
        uint64_t data = 0;
        if (!file.read(reinterpret_cast<char*>(&data), sizeof(data))) {
            break;
        }

        uint16_t op = data & 0x1F;

        if (op == OMNI_RES_REQUEST_OPCODE) {
            uint16_t slaveThreadNum = (data >> 5) & 0x1F;
            uint16_t notifyNumOnMainThread = (data >> 10) & 0x1F;
            uint16_t notifyNumPerThread = (data >> 15) & 0x1F;
            uint16_t netLayer = (data >> 20) & 0x3;
            uint16_t chanCount = (data >> 22) & 0xFF;

            uint16_t blockNumAiv = 1;

            xmlInfo_.resInfo.slaveThreadNum = slaveThreadNum;
            xmlInfo_.resInfo.notifyNumOnMainThread = notifyNumOnMainThread;
            xmlInfo_.resInfo.notifyNumPerThread = notifyNumPerThread;
            xmlInfo_.resInfo.netLayerNum = netLayer;
            xmlInfo_.resInfo.blockNumAiv = blockNumAiv;

            offset += sizeof(data);

            std::map<u32, OmniChannelInfo> mapChannelInfo;
            for (uint16_t i = 0; i < chanCount; i++) {
                file.seekg(offset);
                uint32_t channelData = 0;
                if (!file.read(reinterpret_cast<char*>(&channelData), sizeof(channelData))) {
                    break;
                }

                uint16_t netLayerId = (channelData >> 0) & 0x1F;
                uint16_t localRank = (channelData >> 5) & 0x3FF;
                uint16_t remoteRank = (channelData >> 15) & 0x3FF;
                uint16_t linkProto = (channelData >> 25) & 0x7;
                offset += sizeof(channelData);

                if (localRank != myRank_) {
                    continue;
                }

                OmniChannelInfo omniChannelInfo;
                omniChannelInfo.netlayerId = netLayerId;
                omniChannelInfo.remoteRank = remoteRank;
                omniChannelInfo.channelProtocol = static_cast<CommProtocol>(linkProto);
                mapChannelInfo[remoteRank] = omniChannelInfo;
            }
            xmlInfo_.resInfo.mapchannelInfo.push_back(mapChannelInfo);
        } else if (op == OMNI_PRESYNC_OPCODE || op == OMNI_POSTSYNC_OPCODE) {
            uint16_t subThreadNum = (data >> 10) & 0x1F;
            offset += sizeof(data);
            offset += subThreadNum * sizeof(uint8_t);
        } else {
            uint16_t netlayerId = (data >> 5) & 0x3;
            uint16_t linkProto = (data >> 7) & 0x7;
            uint16_t sliceNum = (data >> 10) & 0x3FF;
            uint16_t srcSliceCnt = (data >> 20) & 0xF;
            uint16_t dstSliceCnt = (data >> 24) & 0xF;
            uint16_t notifyFlag = (data >> 28) & 0x1;
            uint16_t notifyThread = (data >> 29) & 0xF;
            uint16_t waitFlag = (data >> 33) & 0x1;
            uint16_t waitThread = (data >> 34) & 0xF;
            uint16_t threadIdx = (data >> 38) & 0x1F;
            uint16_t reduceType = (data >> 43) & 0x3;
            uint16_t inputDataType = (data >> 45) & 0xF;
            uint16_t outputDataType = (data >> 49) & 0xF;
            uint16_t instructionId = (data >> 53) & 0x3FF;
            (void)linkProto;
            (void)notifyFlag;
            (void)notifyThread;
            (void)waitFlag;
            (void)waitThread;
            (void)instructionId;

            OmniSendRecvInfo omniSendRecvInfo;
            omniSendRecvInfo.optype = static_cast<OpType>(op);
            omniSendRecvInfo.inputDataType = static_cast<HcclDataType>(inputDataType);
            omniSendRecvInfo.outputDataType = static_cast<HcclDataType>(outputDataType);
            omniSendRecvInfo.reduceType = static_cast<HcclReduceOp>(reduceType);
            omniSendRecvInfo.sliceNum = sliceNum;
            omniSendRecvInfo.netlayerId = netlayerId;
            omniSendRecvInfo.threadIdx = threadIdx;

            offset += sizeof(data);

            for (uint16_t i = 0; i < srcSliceCnt; i++) {
                file.seekg(offset);
                uint32_t srcData = 0;
                if (!file.read(reinterpret_cast<char*>(&srcData), sizeof(srcData))) {
                    break;
                }
                offset += sizeof(srcData);

                OmniSliceInfo omniSliceInfo;
                omniSliceInfo.sliceType = static_cast<BufferTypeTmp>((srcData >> 0) & 0x3);
                omniSliceInfo.sliceIdx = (srcData >> 2) & 0x3FF;
                omniSliceInfo.remoteRank = (srcData >> 12) & 0x3FF;
                omniSendRecvInfo.srcSliceInfo.push_back(omniSliceInfo);
            }

            for (uint16_t i = 0; i < dstSliceCnt; i++) {
                file.seekg(offset);
                uint32_t dstData = 0;
                if (!file.read(reinterpret_cast<char*>(&dstData), sizeof(dstData))) {
                    break;
                }
                offset += sizeof(dstData);

                OmniSliceInfo omniSliceInfo;
                omniSliceInfo.sliceType = static_cast<BufferTypeTmp>((dstData >> 0) & 0x3);
                omniSliceInfo.sliceIdx = (dstData >> 2) & 0x3FF;
                omniSliceInfo.remoteRank = (dstData >> 12) & 0x3FF;
                omniSendRecvInfo.dstSliceInfo.push_back(omniSliceInfo);
            }
            xmlInfo_.vecSendRecvInfo.push_back(omniSendRecvInfo);
        }
    } while (file.peek() != EOF);

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
