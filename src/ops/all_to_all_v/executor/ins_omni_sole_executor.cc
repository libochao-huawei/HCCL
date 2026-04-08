/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
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

template <typename AlgTopoMatch, typename InsAlgTemplate>
uint32_t InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ReadBits(std::ifstream& file, uint64_t offset, size_t numBits) {
    uint32_t result = 0;
    uint8_t byte;
    size_t bitsRead = 0;
    size_t bitsLeftInByte = 8; // 每个字节有8位

    file.seekg(offset); // 定位到正确的偏移位置

    while (bitsRead < numBits && file.read(reinterpret_cast<char*>(&byte), 1)) {
        size_t bitsToRead = std::min(numBits - bitsRead, bitsLeftInByte);
        result = (result << bitsToRead) | (byte >> (8 - bitsToRead));
        bitsRead += bitsToRead;
        bitsLeftInByte -= bitsToRead;
        if (bitsLeftInByte == 0) {
            byte = 0; // 重置byte以读取下一个字节的数据
            bitsLeftInByte = 8;
        } else {
            byte &= (1 << bitsLeftInByte) - 1; // 清除已读部分
        }
    }

    // 如果需要，可以右移以对齐高位到uint32_t的最低位
    if (bitsRead < numBits) {
        result >>= (32 - bitsRead); // 右移以对齐高位到最低位
    }

    return result;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ParseXmlInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    std::ifstream file("example.bin", std::ios::binary);
    if (!file) {
        std::cerr << "can not open file" << std::endl;
        return HCCL_E_PARA;
    }
    XmlInfo xmlInfo;
    uint64_t offset = 0;

    // 控制面资源
    uint16_t op = ReadBits(file, offset, 8); // op
    offset += 8;
    uint16_t notifyNumOnMainThread = ReadBits(file, offset, 8); // notifyNumOnMainThread
    offset += 8;
    uint16_t notifyNumPerThread = ReadBits(file, offset, 8); // notifyNumPerThread
    offset += 8;
    uint16_t netLayer = ReadBits(file, offset, 8); // notifyNumPerThread
    offset += 8;
    uint16_t chanCount = ReadBits(file, offset, 8); // ChanCount
    offset += 8;
    std::map<u32, OmniChannelInfo> mapChannelInfo;
    for (uint16_t i = 0; i < chanCount; i++) {
        OmniChannelInfo omniChannelInfo;
        uint16_t channelId  = ReadBits(file, offset, 8); // channelId
        offset += 8;
        uint16_t localRank  = ReadBits(file, offset, 8); // localRank
        offset += 8;
        uint16_t remoteRank  = ReadBits(file, offset, 8); // remoteRank
        offset += 8;
        uint16_t linkProto  = ReadBits(file, offset, 8); // linkProto
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
        // 64bit 操作控制字段
        uint16_t opcode = ReadBits(file, offset, 5); // opcode
        offset += 5;
        uint16_t linkType = ReadBits(file, offset, 2); // LinkType
        offset += 2;
        uint16_t linkProto = ReadBits(file, offset, 3); // LinkProto
        offset += 3;
        uint16_t localRankID = ReadBits(file, offset, 10); // LocalRankID
        offset += 10;
        uint16_t sliceNum = ReadBits(file, offset, 10); // Slice_Num
        offset += 10;
        uint16_t srcSliceCnt = ReadBits(file, offset, 4); // Src_Slice_Cnt
        offset += 4;
        uint16_t dstSliceCnt = ReadBits(file, offset, 4); // Dst_Slice_Cnt
        offset += 4;
        uint16_t notifyFlag = ReadBits(file, offset, 1); // Notify_Flag
        offset += 1;
        uint16_t notifyThread = ReadBits(file, offset, 4); // Notify_Thread
        offset += 4;
        uint16_t waitFlag = ReadBits(file, offset, 1); // Wait_Flag
        offset += 1;
        uint16_t waitThread = ReadBits(file, offset, 4); // Wait_Thread
        offset += 4;
        offset += 16; // reserved

        // 64bit 通道与数据类型
        uint16_t sendChannelID = ReadBits(file, offset, 8); // SendChannelID
        offset += 8;
        uint16_t recvChannelID = ReadBits(file, offset, 8); // RecvChannelID
        offset += 8;
        uint16_t threadIdx = ReadBits(file, offset, 5); // ThreadIdx
        offset += 5;
        uint16_t instructionID = ReadBits(file, offset, 16); // InstructionID
        offset += 16;
        uint16_t reduceType = ReadBits(file, offset, 2); // reduceType
        offset += 2;
        uint16_t inputDataType = ReadBits(file, offset, 4); // inputDataType
        offset += 4;
        uint16_t outputDataType = ReadBits(file, offset, 4); // outputDataType
        offset += 4;
        offset += 17; // reserved

        omniSendRecvInfo.optype = static_cast<OpType>(opcode);
        omniSendRecvInfo.inputDataType = static_cast<HcclDataType>(inputDataType);
        omniSendRecvInfo.outputDataType = static_cast<HcclDataType>(outputDataType);
        omniSendRecvInfo.reduceType = static_cast<HcclReduceOp>(reduceType);
        omniSendRecvInfo.sliceNum = sliceNum;
        omniSendRecvInfo.linkType = linkType;
        omniSendRecvInfo.threadIdx = threadIdx;


        for (uint16_t i = 0; i < srcSliceCnt; i++) {
            OmniSliceInfo omniSliceInfo;
            uint16_t srcBufferType = ReadBits(file, offset, 2); // srcBufferType
            offset += 2;
            uint16_t srcSliceIdx = ReadBits(file, offset, 10); // srcSliceIdx
            offset += 10;
            uint16_t remoteRank = ReadBits(file, offset, 10); // RemoteRank
            offset += 10;
            omniSliceInfo.sliceType = srcBufferType;
            omniSliceInfo.sliceIdx = srcSliceIdx;
            omniSliceInfo.remoteRank = remoteRank;
            omniSendRecvInfo.srcSliceInfo.push_back(omniSliceInfo);
        }

        for (uint16_t i = 0; i < dstSliceCnt; i++) {
            OmniSliceInfo omniSliceInfo;
            uint16_t dstBufferType = ReadBits(file, offset, 2); // dstBufferType
            offset += 2;
            uint16_t dstSliceIdx = ReadBits(file, offset, 10); // dstSliceIdx
            offset += 10;
            uint16_t remoteRank = ReadBits(file, offset, 10); // RemoteRank
            offset += 10;

            omniSliceInfo.sliceType = dstBufferType;
            omniSliceInfo.sliceIdx = dstSliceIdx;
            omniSliceInfo.remoteRank = remoteRank;
            omniSendRecvInfo.dstSliceInfo.push_back(omniSliceInfo);
        }
        xmlInfo.vecSendRecvInfo.push_back(omniSendRecvInfo);

        file.seekg(offset); // 定位到正确的偏移位置
    } while(file.peek() != EOF);

    // // 解析bin文件 读取内容存入xmlInfo结构体中
    // xmlInfo_.resInfo.slaveThreadNum = 3;
    // xmlInfo_.resInfo.notifyNumOnMainThread = 3;
    // xmlInfo_.resInfo.notifyNumPerThread = 3;
    // xmlInfo_.resInfo.netLayerNum = 1;

    // if (myRank_ == 0) {
    //     std::map<u32, OmniChannelInfo> tmp;
    //     OmniChannelInfo tmpInfo;
    //     tmpInfo.channelId = 0;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 1;
    //     tmp[1] = tmpInfo;

    //     tmpInfo.channelId = 1;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 2;
    //     tmp[2] = tmpInfo;

    //     tmpInfo.channelId = 2;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 3;
    //     tmp[3] = tmpInfo;
    //     xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    // } else if (myRank_ == 1) {
    //     std::map<u32, OmniChannelInfo> tmp;
    //     OmniChannelInfo tmpInfo;
    //     tmpInfo.channelId = 0;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 0;
    //     tmp[0] = tmpInfo;

    //     tmpInfo.channelId = 1;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 2;
    //     tmp[2] = tmpInfo;

    //     tmpInfo.channelId = 2;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 3;
    //     tmp[3] = tmpInfo;
    //     xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    // } else if (myRank_ == 2) {
    //     std::map<u32, OmniChannelInfo> tmp;
    //     OmniChannelInfo tmpInfo;
    //     tmpInfo.channelId = 0;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 0;
    //     tmp[0] = tmpInfo;

    //     tmpInfo.channelId = 1;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 1;
    //     tmp[1] = tmpInfo;

    //     tmpInfo.channelId = 2;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 3;
    //     tmp[3] = tmpInfo;
    //     xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    // } else if (myRank_ == 3) {
    //     std::map<u32, OmniChannelInfo> tmp;
    //     OmniChannelInfo tmpInfo;
    //     tmpInfo.channelId = 0;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 0;
    //     tmp[0] = tmpInfo;

    //     tmpInfo.channelId = 1;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 1;
    //     tmp[1] = tmpInfo;

    //     tmpInfo.channelId = 2;
    //     tmpInfo.channelProtocol = COMM_PROTOCOL_HCCS;
    //     tmpInfo.remoteRank = 2;
    //     tmp[2] = tmpInfo;
    //     xmlInfo_.resInfo.mapchannelInfo.push_back(tmp);
    // }

    // if (myRank_ == 0) {        
    //     OmniSendRecvInfo tmpInfo;
    //     tmpInfo.optype = OP_LOCAL_COPY;
    //     tmpInfo.srcSliceInfo.resize(1);
    //     tmpInfo.srcSliceInfo[0].sliceType = 0;
    //     tmpInfo.srcSliceInfo[0].sliceIdx = 0;

    //     tmpInfo.dstSliceInfo.resize(1);
    //     tmpInfo.dstSliceInfo[0].sliceType = 1;
    //     tmpInfo.dstSliceInfo[0].sliceIdx = 0;
    //     tmpInfo.remoteRank = 1;

    //     tmpInfo.sliceNum = 4;

    //     xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    // } else if (myRank_ == 1) {        
    //     OmniSendRecvInfo tmpInfo;
    //     tmpInfo.optype = OP_LOCAL_COPY;
    //     tmpInfo.srcSliceInfo.resize(1);
    //     tmpInfo.srcSliceInfo[0].sliceType = 0;
    //     tmpInfo.srcSliceInfo[0].sliceIdx = 1;

    //     tmpInfo.dstSliceInfo.resize(1);
    //     tmpInfo.dstSliceInfo[0].sliceType = 1;
    //     tmpInfo.dstSliceInfo[0].sliceIdx = 1;
    //     tmpInfo.remoteRank = 1;
    //     tmpInfo.sliceNum = 4;

    //     xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    // } else if (myRank_ == 2) {        
    //     OmniSendRecvInfo tmpInfo;
    //     tmpInfo.optype = OP_LOCAL_COPY;
    //     tmpInfo.srcSliceInfo.resize(1);
    //     tmpInfo.srcSliceInfo[0].sliceType = 0;
    //     tmpInfo.srcSliceInfo[0].sliceIdx = 2;

    //     tmpInfo.dstSliceInfo.resize(1);
    //     tmpInfo.dstSliceInfo[0].sliceType = 1;
    //     tmpInfo.dstSliceInfo[0].sliceIdx = 2;
    //     tmpInfo.remoteRank = 1;
    //     tmpInfo.sliceNum = 4;

    //     xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    // } else if (myRank_ == 3) {        
    //     OmniSendRecvInfo tmpInfo;
    //     tmpInfo.optype = OP_LOCAL_COPY;
    //     tmpInfo.srcSliceInfo.resize(1);
    //     tmpInfo.srcSliceInfo[0].sliceType = 0;
    //     tmpInfo.srcSliceInfo[0].sliceIdx = 3;

    //     tmpInfo.dstSliceInfo.resize(1);
    //     tmpInfo.dstSliceInfo[0].sliceType = 1;
    //     tmpInfo.dstSliceInfo[0].sliceIdx = 3;
    //     tmpInfo.remoteRank = 1;
    //     tmpInfo.sliceNum = 4;

    //     xmlInfo_.vecSendRecvInfo.push_back(tmpInfo);
    // }
    


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
    }
    templateAlgRes.threads = resCtx.threads;

    //计算loop  ccu 不用cclbuff，根据UB_MAX_DATA_SIZE来计算
    u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_INFO("[InsOmniSoleExecutor][OrchestrateLoop]loopTimes = [%u]", loopTimes);

    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_ / xmlInfo_.vecSendRecvInfo[0].sliceNum;
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

}