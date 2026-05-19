/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_resource_graph_mode.h"
#include "hccl/hcom.h"
#include <cstddef>
#include <cstring>
#include "hcom.h"
#include "op_common.h"

HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode **opParam)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void**转换为OpParamGraphMode**
    OpParamGraphMode **paramPtr = reinterpret_cast<OpParamGraphMode **>(opParam);
    *paramPtr = new OpParamGraphMode();
    if (*paramPtr == nullptr) {
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode *opParam)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    delete paramPtr;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode *opParam, const char *opType, HcclCMDType cmdType)
{
    if (opParam == nullptr || opType == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    strncpy_s(paramPtr->opType, sizeof(paramPtr->opType), opType, sizeof(paramPtr->opType) - 1);
    paramPtr->opTypeAiv = cmdType;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeDataCount(OpParamGraphMode *opParam, const u64 *dataCount)
{
    if (opParam == nullptr || dataCount == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    memcpy_s(&paramPtr->dataCount, sizeof(paramPtr->dataCount), dataCount, sizeof(u64));
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeDataType(OpParamGraphMode *opParam, const HcclDataType dataType)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    paramPtr->dataType = dataType;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeRankSize(OpParamGraphMode *opParam, const u32 *rankSize)
{
    if (opParam == nullptr || rankSize == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    memcpy_s(&paramPtr->rankSize, sizeof(paramPtr->rankSize), rankSize, sizeof(u32));
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeHCCLBufferSize(OpParamGraphMode *opParam, const u64 *hcclBufferSize)
{
    if (opParam == nullptr || hcclBufferSize == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    memcpy_s(&paramPtr->hcclBufferSize, sizeof(paramPtr->hcclBufferSize), hcclBufferSize, sizeof(u64));
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeGroup(OpParamGraphMode *opParam, const char *group)
{
    if (opParam == nullptr || group == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    strncpy_s(paramPtr->group, sizeof(paramPtr->group), group, sizeof(paramPtr->group) - 1);
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeReduceOp(OpParamGraphMode *opParam, HcclReduceOp reduceOp);
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    paramPtr->op = reduceOp;
    return HCCL_SUCCESS;
}

HcclResult HcclSetAivSelectOpParamGraphMode(OpParamGraphMode *opParam, u32 aivCoreLimit)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    paramPtr->aivCoreLimit = aivCoreLimit;
    return HCCL_SUCCESS;
}

HcclResult HcclCalcOpResOnlineGraphMode(OpParamGraphMode *opParam, u64 *opMemSize, u32 *streamNum, u32 *taskNum, u32 *aivCoreNum)
{
    HCCL_INFO("Enter HcclCalcOpResOnlineGraphMode.");
    CHK_RET(CheckCalcResInputGraphMode(opParam, opMemSize, streamNum, taskNum, aivCoreNum));
    // 将void**转换为OpParamGraphMode**
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    if (paramPtr == nullptr) {
        return HCCL_E_PARA;
    }
    // 为了兼容，创建临时的 ResResponseGraphMode 结构
    ResResponseGraphMode resResponse = {0, 0, 0, 0};
    HCCL_INFO("Start to calc op resource online.");

    bool ifAiv = false;
    char algName[ALG_NAME_MAX_LEN];
    char *pAlgName = algName;
 	CHK_RET(HcclSelectAlgGraphMode(opParam->group, opParam->dataCount, opParam->dataType, opParam->op,
        opParam->opTypeAiv, opParam->aivCoreLimit, &ifAiv, &pAlgName));
    if (ifAiv) {
        // aiv引擎计算资源
        ops_hccl::HcclCalcAivResOffline(&resResponse, paramPtr);
    } else {
        // aicpu引擎计算资源
        ops_hccl::HcclCalcAicpuResOffline(&resResponse);

        // ccu引擎计算资源
        ops_hccl::HcclCalcCcuResOffline(opParam, &resResponse);
    }

    // 将结果复制到输出参数
    *opMemSize = resResponse.opMemSize;
    *streamNum = resResponse.streamNum;
    *taskNum = resResponse.taskNum;
    *aivCoreNum = resResponse.aivCoreNum;

    return HCCL_SUCCESS;
}

HcclResult HcclCalcOpResOfflineGraphMode(OpParamGraphMode *opParam, u64 *opMemSize, u32 *streamNum, u32 *taskNum, u32 *aivCoreNum)
{
    HCCL_INFO("Enter HcclCalcOpResOfflineGraphMode.");
    CHK_RET(CheckCalcResInputGraphMode(opParam, opMemSize, streamNum, taskNum, aivCoreNum));
    // 将void**转换为OpParamGraphMode**
    OpParamGraphMode *paramPtr = reinterpret_cast<OpParamGraphMode *>(opParam);
    if (paramPtr == nullptr) {
        return HCCL_E_PARA;
    }
    // 为了兼容，创建临时的 ResResponseGraphMode 结构
    ResResponseGraphMode resResponse = {0, 0, 0, 0};
    HCCL_INFO("Start to calc op resource offline.");

    bool ifAiv = false;
    char algName[ALG_NAME_MAX_LEN];
    char *pAlgName = algName;
 	CHK_RET(HcclSelectAlgGraphMode(opParam->group, opParam->dataCount, opParam->dataType, opParam->op,
        opParam->opTypeAiv, opParam->aivCoreLimit, &ifAiv, &pAlgName));
    if (ifAiv) {
        // aiv引擎计算资源
        ops_hccl::HcclCalcAivResOffline(&resResponse, paramPtr);
    } else {
        // aicpu引擎计算资源
        ops_hccl::HcclCalcAicpuResOffline(&resResponse);

        // ccu引擎计算资源
        ops_hccl::HcclCalcCcuResOffline(opParam, &resResponse);
    }

    // 将结果复制到输出参数
    *opMemSize = resResponse.opMemSize;
    *streamNum = resResponse.streamNum;
    *taskNum = resResponse.taskNum;
    *aivCoreNum = resResponse.aivCoreNum;

    return HCCL_SUCCESS;
}

HcclResult HcclSetAivCoreLimitGraphMode(const char *group, u32 aivCoreLimit)
{
    if (group == nullptr) {
        HCCL_ERROR("[HcclSetAivCoreLimitGraphMode] group is nullptr");
        return HCCL_E_PARA;
    }

    ops_hccl::AivParamStorage *aivParam = nullptr;
    CHK_RET(ops_hccl::GetAivParamStorage(group, &aivParam));

    aivParam->aivCoreLimit = aivCoreLimit;

    HCCL_INFO("[HcclSetAivCoreLimitGraphMode] Set aivCoreLimit[%u] for group[%s]", aivCoreLimit, group);

    return HCCL_SUCCESS;
}

HcclResult HcclSelectAlgGraphMode(const char *group, u64 count, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType,
                        u32 aivCoreLimit, bool *ifAiv, char **algName)
{
    HCCL_INFO("[HcclSelectAlgGraphMode] Start.");
    
    if (group == nullptr || ifAiv == nullptr || algName == nullptr) {
        HCCL_ERROR("[HcclSelectAlgGraphMode] Invalid parameters");
        return HCCL_E_PARA;
    }
    
    HcclComm hcclComm = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(group, &hcclComm));
    
    CHK_RET(ops_hccl::InitEnvConfig());
    
    ops_hccl::OpParam param;
    CHK_RET(HcclGetCommName(hcclComm, param.commName));
    
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    
    param.opType = opType;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.reduceType = op;
    param.opMode = ops_hccl::OpMode::OFFLOAD;
    param.numBlocksLimit = aivCoreLimit;
    param.enableDetour = false;
    param.deviceType = deviceType;

    int ret = sprintf_s(param.tag, sizeof(param.tag), "SelectAlg_%d_%s", static_cast<int>(opType), param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[HcclSelectAlgGraphMode] failed to fill param.tag"), HCCL_E_INTERNAL);
    
    CHK_RET(ops_hccl::HcclGetOpExpansionMode(hcclComm, param));

    std::unique_ptr<ops_hccl::TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<ops_hccl::TopoInfoWithNetLayerDetails>();
    std::string localAlgName;
    CHK_RET(ops_hccl::Selector(hcclComm, param, topoInfo, localAlgName));
    
    *ifAiv = (param.engine == CommEngine::COMM_ENGINE_AIV);
    
    // 分配内存并拷贝字符串
    *algName = (char*)malloc(localAlgName.size() + 1);
    if (*algName == nullptr) {
        HCCL_ERROR("[HcclSelectAlgGraphMode] malloc failed for algName");
        return HCCL_E_INTERNAL;
    }
    strncpy_s(*algName, localAlgName.size() + 1, localAlgName.c_str(), localAlgName.size());
    
    HCCL_INFO("[HcclSelectAlgGraphMode] Success. ifAiv=%d, algName=%s", *ifAiv, *algName);
    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult HcclCalcAicpuResOffline(ResResponseGraphMode *resResponse)
{
    if (resResponse == nullptr) {
        return HCCL_E_PARA;
    }
    u64 aicpuOpMemSize = 0;
    u32 aicpuStreamNum = 0;
    u32 aicpuTaskNum = 3;

    resResponse->opMemSize = std::max(resResponse->opMemSize, aicpuOpMemSize);
    resResponse->streamNum = std::max(resResponse->streamNum, aicpuStreamNum);
    resResponse->taskNum = std::max(resResponse->taskNum, aicpuTaskNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcAivResOffline(ResResponseGraphMode *resResponse, OpParamGraphMode *paramPtr)
{
    if (resResponse == nullptr || paramPtr == nullptr || paramPtr->aivCoreLimit == 0) {
        return HCCL_E_PARA;
    }

    resResponse->opMemSize = 0;
    resResponse->streamNum = 0;
    resResponse->taskNum = 3;
    resResponse->aivCoreNum = paramPtr->aivCoreLimit;
    HCCL_INFO("[HcclCalcAivResOffline] opMemSize[%llu], streamNum[%u], taskNum[%u], aivCoreNum[%u]",
        resResponse->opMemSize, resResponse->streamNum, resResponse->taskNum, resResponse->aivCoreNum);
    return HCCL_SUCCESS;
}

HcclResult CheckCalcResInputGraphMode(const OpParamGraphMode *opParam, const u64 *opMemSize, const u32 *streamNum, 
                                      const u32 *taskNum, const u32 *aivCoreNum)
{
    CHK_PTR_NULL(opParam);
    CHK_PTR_NULL(opMemSize);
    CHK_PTR_NULL(streamNum);
    CHK_PTR_NULL(taskNum);
    CHK_PTR_NULL(aivCoreNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcCcuResOffline(OpParamGraphMode *opParam, ResResponseGraphMode *resResponse)
{
    HCCL_INFO("Entry HcclCalcCcuResOffline.");
    if (resResponse == nullptr || opParam == nullptr) {
        return HCCL_E_PARA;
    }

    // ccu的资源申请
    u64 ccuOpMemSize = 0;
    u32 ccuStreamNum = 3;
    u32 ccuTaskNum = 0;

    CHK_PRT(CalcTaskNum(opParam, ccuTaskNum));

    resResponse->opMemSize = std::max(resResponse->opMemSize, ccuOpMemSize);
    resResponse->streamNum = std::max(resResponse->streamNum, ccuStreamNum);
    resResponse->taskNum = std::max(resResponse->taskNum, ccuTaskNum);
    HCCL_INFO("[HcclCalcCcuResOffline] opMemSize[%llu], streamNum[%llu], taskNum[%llu]", resResponse->opMemSize, resResponse->streamNum, resResponse->taskNum);
    return HCCL_SUCCESS;
}

HcclResult CalcTaskNum(OpParamGraphMode *opParam, u32 &ccuTaskNum)
{
    HCCL_INFO("[CalcTaskNum] begin");
    if (opParam->hcclBufferSize == 0 || opParam->rankSize == 0) {
        ccuTaskNum = GE_PARALLEL;
        return HCCL_SUCCESS;
    }

    u64 dataCount = opParam->dataCount;
    u64 rankSize = opParam->rankSize;
    u64 scratchBufferSize = opParam->hcclBufferSize;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 dataType = opParam->dataType;
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 maxDataSizePerLoop;
    u64 maxDataCountPerLoop;
    u64 loopTimes;
    HCCL_INFO("[CalcTaskNum] opType[%s] scratchBufferSize[%llu] dataCount[%llu] rankSize[%llu]", 
            opParam->opType, scratchBufferSize, dataCount, rankSize);
    if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLTOALL) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize / rankSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLTOALLV || opParam->opType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
        ccuTaskNum = 1;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_REDUCE) { 
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        u64 scratchBoundDataSize = scratchBufferSize / rankSize / 128 * 128;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_SCATTER) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLGATHERV) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    }
    HCCL_INFO("[CalcTaskNum] maxDataSizePerLoop[%llu] maxDataCountPerLoop[%llu] loopTimes[%llu] ccuTaskNum[%llu]", 
            maxDataSizePerLoop, maxDataCountPerLoop, loopTimes, ccuTaskNum);
    HCCL_INFO("[CalcTaskNum] end.");
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
