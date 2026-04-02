/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mc2.h"
#include "log.h"
#include "sal.h"
#include "alg_env_config.h"
#include "hccl_inner.h"
#include "param_check.h"

using namespace ops_hccl;

constexpr uint32_t ALG_CONFIG_SIZE = 128;
struct HcclOpArgs {
    HcclDataType srcDataType;
    HcclDataType dstDataType;
    HcclReduceOp reduceType;
    uint64_t count;
    char algConfig[ALG_CONFIG_SIZE];
    CommEngine commEngine;
    uint64_t reverse;

    void Init() {
        srcDataType = HCCL_DATA_TYPE_FP16;
        dstDataType = HCCL_DATA_TYPE_FP16;
        reduceType = HCCL_REDUCE_SUM;
        count = 0;
    }
};

HcclResult HcclKfcAllocOpArgs(void **opArgs)
{
    CHK_PTR_NULL(opArgs);

    HcclOpArgs *opArgsMem = (HcclOpArgs *)malloc(sizeof(HcclOpArgs));
    if (opArgsMem == nullptr) {
        HCCL_ERROR("[HcclKfcAllocOpArgs] malloc HcclOpArgs mem failed, please check.");
        return HCCL_E_INTERNAL;
    }
    opArgsMem->Init();
    *opArgs = opArgsMem;
    HCCL_RUN_INFO("[HcclKfcAllocOpArgs] malloc HcclOpArgs success, please fill mem[%p->%p] in it.", opArgs, *opArgs);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcFreeOpArgs(void *opArgs)
{
    CHK_PTR_NULL(opArgs);

    free(opArgs);
    opArgs = nullptr;

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetSrcDataType(void *opArgs, uint8_t srcDataType)
{
    CHK_PTR_NULL(opArgs);
    CHK_RET(HcomCheckDataType(static_cast<HcclDataType>(srcDataType)));

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->srcDataType = static_cast<HcclDataType>(srcDataType);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetDstDataType(void *opArgs, uint8_t dstDataType)
{
    CHK_PTR_NULL(opArgs);
    CHK_RET(HcomCheckDataType(static_cast<HcclDataType>(dstDataType)));

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->dstDataType = static_cast<HcclDataType>(dstDataType);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetReduceType(void *opArgs, uint32_t reduceType)
{
    CHK_PTR_NULL(opArgs);
    CHK_RET(HcomCheckReductionOp(static_cast<HcclReduceOp>(reduceType)));

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->reduceType = static_cast<HcclReduceOp>(reduceType);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetCount(void *opArgs, uint64_t count)
{
    CHK_PTR_NULL(opArgs);
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[%s] count[%llu] is invalid (bigger than MAX count[%lu])", __func__, count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->count = count;

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetAlgConfig(void *opArgs, char *algConfig)
{
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(algConfig);

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    s32 ret = strcpy_s(opArgsPtr->algConfig, ALG_CONFIG_SIZE, algConfig);
    if (ret != EOK) {
        HCCL_ERROR("[%s] strcpy_s algConfig failed, ret[%d]", __func__, ret);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetCommEngine(void *opArgs, uint8_t commEngine)
{
    CHK_PTR_NULL(opArgs);
    // A3只支持AICPU和AIV场景
    if (commEngine != COMM_ENGINE_AICPU && commEngine != COMM_ENGINE_AIV) {
        HCCL_ERROR("[%s] commEngine[%u] not supported", __func__, commEngine);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->commEngine = static_cast<CommEngine>(commEngine);

    return HCCL_SUCCESS;
}

HcclResult HcclCreateOpResCtx(HcclComm comm, uint8_t opType, void *opArgs, void **opResCtx)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(opResCtx);
    if (opType >= static_cast<uint8_t>(HcclCMDType::HCCL_CMD_MAX)) {
        HCCL_ERROR("[%s] invalid opType[%u]", __func__, opType);
        return HCCL_E_PARA;
    }

    CHK_RET(InitEnvConfig());

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcCreateOpResCtx, opType[%u], opArgs[%p], srcDataType[%u], dstDataType[%u], reduceType[%u], "
            "count[%llu], algConfig[%s], commEngine[%u], opResCtx[%p]",
            opType, opArgs, opArgsPtr->srcDataType, opArgsPtr->dstDataType, opArgsPtr->reduceType,
            opArgsPtr->count, opArgsPtr->algConfig, opArgsPtr->commEngine, opResCtx);
    }

    CHK_RET(HcclCreateOpResCtxInner(comm, opType, opArgsPtr->srcDataType, opArgsPtr->dstDataType,
        opArgsPtr->reduceType, opArgsPtr->count, opArgsPtr->algConfig, opArgsPtr->commEngine, opResCtx));

    return HCCL_SUCCESS;
}

constexpr uint32_t INIT_TILING_VERSION = 100U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;
struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};


HcclResult CheckInputParam(const HcclComm comm, const void* mc2Tiling, const aclrtStream stream)
{   
    // 检查comm是否为空指针
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllocComResourceByTiling", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    
    // 检查sendBuf是否为空指针
    RPT_INPUT_ERR(mc2Tiling == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllocComResourceByTiling", "nullptr", "mc2Tiling", "non-null pointer"}));
    CHK_PTR_NULL(mc2Tiling);
    
    // 检查stream是否为空指针
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclAllocComResourceByTiling", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    return HCCL_SUCCESS;
}

HcclResult HcclGetTilingList(const void *mc2Tiling, const void *p[], uint32_t &cnt)
{
    const u32 *versionPtr = static_cast<const u32 *>(mc2Tiling);
    const u32 version = *(versionPtr++);
    CHK_PRT_RET(version < MC2_TILING_VERSION, HCCL_ERROR("Invalid tiling version %u.", version), HCCL_E_PARA);

    cnt = *(versionPtr++);
    CHK_PRT_RET(cnt > MAX_HCOM_NUM, HCCL_ERROR("Invalid hcom tiling number %u.", cnt), HCCL_E_PARA);

    u64 serverCfgAddr = reinterpret_cast<u64>(versionPtr) + sizeof(Mc2ServerCfg);
    for (uint32_t i = 0U; i < MAX_CC_TILING_NUM; ++i) {
        p[i] = reinterpret_cast<const void *>(reinterpret_cast<const u8 *>(mc2Tiling) + versionPtr[i]);
    }
    HCCL_INFO("HcclGetTilingList version[%u] cnt[%u]", version, cnt);
    return HCCL_SUCCESS;
}

HcclResult CheckIsReduce(const Mc2CcTilingInner *ccTiling, bool *isReduce)
{
    if (ccTiling->opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || ccTiling->opType == HcclCMDType::HCCL_CMD_REDUCE ||
        ccTiling->opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        *isReduce = true;
    } else {
        *isReduce = false;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void* mc2Tiling, void** opResCtx)
{
    HCCL_INFO("Start to run execute HcclReduceScatter");
    // 记录开始时间，用于性能统计
    HcclUs startut = TIME_NOW();
    // 获取设备类型
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    
    // 检查设备类型是否支持新流程，950或910_95支持新流程，其他设备走老流程
    if (deviceType != DevType::DEV_TYPE_950) {
        HCCL_ERROR("[%s] invalid deviceType[%u]", __func__, deviceType);
        return HCCL_E_NOT_SUPPORT;
    }

    // 初始化环境变量配置，解析HCCL相关的环境变量
    // 包括算子展开模式、确定性计算、通信方式、日志开关等配置
    CHK_RET(InitEnvConfig());
    
    // 检查输入参数的合法性（comm、sendBuf、recvBuf、stream不能为空）
    CHK_RET(CheckInputParam(comm, mc2Tiling, stream));
    
    // 获取通信域中的rank数量
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    
    // 获取当前rank的ID
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    
    // 获取通信域名称
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));

    const void *ccTilingList[MAX_HCOM_NUM];
    uint32_t tilingNum;
    CHK_RET(HcclGetTilingList(mc2Tiling, ccTilingList, tilingNum));

    // 构造操作标签，用于日志、错误追踪、topo资源管理
    // topoTag = ccTilingList->opType + commName
    // ctxTag = ccTilingList->groupName + "_" + ccTilingList[0]->opType + "_" + ccTilingList[0]->algConfig + "_" + ccTilingList[0]->commEngine
    std::string topoTag[MAX_CC_TILING_NUM];
    std::string ctxTag;
    for (uint32_t i = 0U; i < tilingNum; ++i) {
        const Mc2CcTilingInner *ccTiling = static_cast<const Mc2CcTilingInner *>(ccTilingList[i]);
        topoTag[i] = std::to_string(ccTiling->opType) + std::string(commName);
        // 检查标签的合法性
        CHK_RET(HcclCheckTag(topoTag[i].c_str()));
        // 检查是否为reduce类型
        bool isReduce;
        CHK_RET(CheckIsReduce(ccTiling, &isReduce));
        // 检查数据类型的合法性
        CHK_RET(CheckDataType(ccTiling->srcDataType, isReduce));

        if (i == 0) {
            ctxTag = std::string(ccTiling->groupName) + "_" + std::string(ccTiling->opType) + "_" + std::string(ccTiling->algConfig) + "_" + std::string(ccTiling->commEngine);
        } else {
            ctxTag += "_" + std::string(ccTiling->opType) + "_" + std::string(ccTiling->algConfig) + "_" + std::string(ccTiling->commEngine);
        }
    }

    // TODO:记录接口入口日志，包含所有关键参数信息

    // 检查userRank是否在有效范围内
    CHK_RET(HcomCheckUserRank(rankSize, userRank));

    OpParam opParam[tilingNum];
    for (uint32_t i = 0U; i < tilingNum; ++i) {
        // TODO: 根据topoTag[i] 获取opParam[i]的参数
    }

    // TODO: 根据ctxTag 申请通信资源

    // 记录退出日志和性能统计信息
    CHK_RET(LogHcclExit("HcclAllocComResourceByTiling", tag, startut));

}