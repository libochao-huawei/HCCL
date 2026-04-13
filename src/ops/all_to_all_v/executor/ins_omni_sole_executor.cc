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
HcclResult InsOmniSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ParseXmlInfo(const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    std::string fileName;
    if (myRank_ == 0) {
        fileName = "rank_0.bin";
    } else if (myRank_ == 1) {
        fileName = "rank_1.bin";
    } else if (myRank_ == 2) {
        fileName = "rank_2.bin";
    } else if (myRank_ == 3) {
        fileName = "rank_3.bin";
    }

    std::ifstream file(fileName, std::ios::binary);
    if (!file) {
        HCCL_INFO("catn not open file %s", fileName.c_str());
        return HCCL_E_PARA;
    }

    uint64_t offset = 0;
    do {
        // 读取64位数据
        uint64_t data;
        file.seekg(offset); // 定位到正确的偏移位置
        file.read(reinterpret_cast<char*>(&data), sizeof(data));
        if (!file) {
            HCCL_INFO("catn not open file %s", fileName.c_str());
            return HCCL_E_PARA;
        }

        HCCL_INFO("111 rank [%u] abb data[%x]", myRank_, data);

        uint16_t op = data & 0x1F;
        HCCL_INFO("111 rank [%u] abc op[%u]", myRank_, op);

        if (op == 0) { // resRequest
            ResRequest resRequest;
            resRequest.opCode = op;
            resRequest.slave = (data >> 5) & 0x1F;
            resRequest.notifyNumOnMainThread = (data >> 10) & 0x1F;
            resRequest.notifyNumPerThread = (data >> 15) & 0x1F;
            resRequest.netLayerNum = (data >> 20) & 0x03; // 2 bits
            resRequest.chanCount = (data >> 22) & 0xFF;   // 8 bits

            xmlInfo_.resInfo.slaveThreadNum = resRequest.slave;
            xmlInfo_.resInfo.notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
            xmlInfo_.resInfo.notifyNumPerThread = resRequest.notifyNumPerThread;
            xmlInfo_.resInfo.netLayerNum = resRequest.netLayerNum;

            HCCL_INFO("rank [%u] slave [%u] notifyNumOnMainThread [%u] notifyNumPerThread [%u] netLayer [%u]  chanCount[%u]", 
                myRank_, resRequest.slave, resRequest.notifyNumOnMainThread, resRequest.notifyNumPerThread, resRequest.netLayerNum, resRequest.chanCount);

            offset += sizeof(data);
            HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);

            Channel channel;
            std::map<u32, OmniChannelInfo> mapChannelInfo;
            for (uint16_t i = 0; i < resRequest.chanCount; i++) {
                file.seekg(offset); // 定位到正确的偏移位置

                uint32_t channelData;
                file.read(reinterpret_cast<char*>(&channelData), sizeof(channelData));
                if (!file) {
                    HCCL_INFO("catn not open file %s", fileName.c_str());
                    return HCCL_E_PARA;
                }

                HCCL_INFO("111 rank [%u] channelData[%x]", myRank_, channelData);
                
                channel.netlayerId = (channelData >> 0) & 0x1F; // 5 bits
                channel.localRank = (channelData >> 5) & 0x3FF; // 10 bits
                channel.remoteRank = (channelData >> 15) & 0x3FF; // 10 bits
                channel.linkProto = (channelData >> 25) & 0x07; // 7 bits

                if (channel.localRank != myRank_) {
                    offset += sizeof(channelData);
                    HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);
                    continue;
                }

                OmniChannelInfo omniChannelInfo;
                omniChannelInfo.netlayerId = channel.netlayerId;
                omniChannelInfo.remoteRank = channel.remoteRank;
                omniChannelInfo.channelProtocol = static_cast<CommProtocol>(channel.linkProto);

                HCCL_INFO("rank [%u] channel [%u] netlayerId [%u] remoteRank[%u] channelProtocol[%u]", 
                    myRank_, i, channel.netlayerId, channel.remoteRank, channel.linkProto);

                mapChannelInfo[channel.remoteRank] = omniChannelInfo;

                offset += sizeof(channelData);
                HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);
            }

            xmlInfo_.resInfo.mapchannelInfo.push_back(mapChannelInfo);
            HCCL_INFO("rank [%u] xmlInfo_.resInfo.mapchannelInfo size[%u]", myRank_, xmlInfo_.resInfo.mapchannelInfo.size());
        } else if (op == 1 || op == 2) { // PreSyncInterThreads PostSyncInterThreads
            offset += 8; // ccu用不到，所以这边不解析，跳过64位
            HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);
            HCCL_INFO("rank [%u] op [%u]", myRank_, op);
        } else {
            file.seekg(offset); // 定位到正确的偏移位置

            uint64_t ctrlData;
            file.read(reinterpret_cast<char*>(&ctrlData), sizeof(ctrlData));
            if (!file) {
                HCCL_INFO("catn not open file %s", fileName.c_str());
                return HCCL_E_PARA;
            }

            HCCL_INFO("111 rank [%u] ctrlData[%x]", myRank_, ctrlData);

            CtrlOp ctrlOp;
            ctrlOp.opCode = (ctrlData >> 0) & 0x1F; // 5 bits
            ctrlOp.netlayerId = (ctrlData >> 5) & 0x03; // 2 bits
            ctrlOp.linkProto = (ctrlData >> 7) & 0x07; // 3 bits
            ctrlOp.sliceNum = (ctrlData >> 10) & 0x3FF; // 10 bits
            ctrlOp.srcSliceNum = (ctrlData >> 20) & 0xF; // 4 bits
            ctrlOp.dstSliceNum = (ctrlData >> 24) & 0xF; // 4 bits
            ctrlOp.notifyFlag = (ctrlData >> 28) & 0x1; // 1 bits
            ctrlOp.notifyThread = (ctrlData >> 29) & 0xF; // 4 bits
            ctrlOp.waitFlag = (ctrlData >> 33) & 0x1; // 1 bits
            ctrlOp.waitThread = (ctrlData >> 34) & 0xF; // 4 bits
            ctrlOp.threadIdx = (ctrlData >> 38) & 0x1F; // 5 bits
            ctrlOp.reduceType = (ctrlData >> 43) & 0x03; // 2 bits
            ctrlOp.inputDataType = (ctrlData >> 45) & 0xF; // 4 bits
            ctrlOp.outputDataType = (ctrlData >> 49) & 0xF; // 4 bits
            ctrlOp.instructionId = (ctrlData >> 53) & 0x3FF; // 10 bits

            offset += sizeof(ctrlData);
            HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);

            OmniSendRecvInfo omniSendRecvInfo;
            omniSendRecvInfo.optype = static_cast<OpType>(ctrlOp.opCode);
            omniSendRecvInfo.inputDataType = static_cast<HcclDataType>(ctrlOp.inputDataType);
            omniSendRecvInfo.outputDataType = static_cast<HcclDataType>(ctrlOp.outputDataType);
            omniSendRecvInfo.reduceType = static_cast<HcclReduceOp>(ctrlOp.reduceType);
            omniSendRecvInfo.sliceNum = ctrlOp.sliceNum;
            omniSendRecvInfo.threadIdx = ctrlOp.threadIdx;
            omniSendRecvInfo.netlayerId = ctrlOp.netlayerId;
            HCCL_INFO("rank [%u] op [%u] sliceNum [%u] netlayerId [%u] threadIdx [%u]", 
                myRank_, omniSendRecvInfo.optype, omniSendRecvInfo.sliceNum, omniSendRecvInfo.netlayerId, omniSendRecvInfo.threadIdx);

            

            for (uint16_t i = 0; i < ctrlOp.srcSliceNum; i++) {
                OmniSliceInfo omniSliceInfo;
                file.seekg(offset); // 定位到正确的偏移位置

                uint32_t srcData;
                file.read(reinterpret_cast<char*>(&srcData), sizeof(srcData));
                if (!file) {
                    HCCL_INFO("catn not open file %s", fileName.c_str());
                    return HCCL_E_PARA;
                }

                HCCL_INFO("111 rank [%u] srcData[%x]", myRank_, srcData);

                SrcSlice srcSlice;
                srcSlice.bufferType = (srcData >> 0) & 0x03; // 2 bits
                srcSlice.sliceIdx = (srcData >> 2) & 0x3FF; // 10 bits
                srcSlice.rankId = (srcData >> 12) & 0x3FF; // 10 bits
                
                omniSliceInfo.sliceType = static_cast<BufferTypeTmp>(srcSlice.bufferType);
                omniSliceInfo.sliceIdx = srcSlice.sliceIdx;
                omniSliceInfo.remoteRank = srcSlice.rankId;
                omniSendRecvInfo.srcSliceInfo.push_back(omniSliceInfo);
                HCCL_INFO("rank [%u] srcSlice srcBufferType [%u] sliceIdx [%u] rankId [%u]", 
                    myRank_, omniSliceInfo.sliceType, omniSliceInfo.sliceIdx, omniSliceInfo.remoteRank);

                offset += sizeof(srcData);
                HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);
                
            }

            for (uint16_t i = 0; i < ctrlOp.dstSliceNum; i++) {
                OmniSliceInfo omniSliceInfo;
                file.seekg(offset); // 定位到正确的偏移位置

                uint32_t dstData;
                file.read(reinterpret_cast<char*>(&dstData), sizeof(dstData));
                if (!file) {
                    HCCL_INFO("catn not open file %s", fileName.c_str());
                    return HCCL_E_PARA;
                }

                HCCL_INFO("111 rank [%u] dstData[%x]", myRank_, dstData);

                DstSlice dstSlice;
                dstSlice.bufferType = (dstData >> 0) & 0x03; // 2 bits
                dstSlice.sliceIdx = (dstData >> 2) & 0x3FF; // 10 bits
                dstSlice.rankId = (dstData >> 12) & 0x3FF; // 10 bits
                
                omniSliceInfo.sliceType = static_cast<BufferTypeTmp>(dstSlice.bufferType);
                omniSliceInfo.sliceIdx = dstSlice.sliceIdx;
                omniSliceInfo.remoteRank = dstSlice.rankId;
                omniSendRecvInfo.dstSliceInfo.push_back(omniSliceInfo);
                HCCL_INFO("rank [%u] dstSlice srcBufferType [%u] sliceIdx [%u] rankId [%u]", 
                    myRank_, omniSliceInfo.sliceType, omniSliceInfo.sliceIdx, omniSliceInfo.remoteRank);

                offset += sizeof(dstData);
                HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);
            }
            xmlInfo_.vecSendRecvInfo.push_back(omniSendRecvInfo);
        }
        
        HCCL_INFO("rank [%u] aaa offset[%u]", myRank_, offset);
    } while(file.peek() != EOF);

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
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_ / xmlInfo_.vecSendRecvInfo[0].sliceNum * 4;
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