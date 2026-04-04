/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALLOC_CTX_RES_H
#define HCCL_ALLOC_CTX_RES_H

#include "topo_host.h"
#include <vector>

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

struct AlgInfo {
    uint64_t offset;
    uint64_t opParam;
};

struct OpResCtx {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint64_t rankId;
    uint64_t rankSize;
    AlgInfo algInfo[MAX_CC_TILING_NUM];
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

HcclResult CheckCommEngine(const void *ccTilingList, uint32_t tilingNum)
{
    for (uint32_t i = 0U; i < tilingNum; ++i) {
        if (ccTilingList[i]->commEngine != static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU)) {
            HCCL_ERROR("Invalid commEngine %u.", ccTilingList[i]->commEngine);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAllocOpResCtx(HcclComm comm, const std::string &ctxTag, const std::vector<OpParam> &opParamVec, void* mc2Tiling, const void *ccTilingList, void** opResCtxPtr)
{
    OpResCtx resCtx;
    Mc2InitTilingInner *initTiling = static_cast<Mc2InitTilingInner *>(mc2Tiling);

    // 1. 申请存放OpParam的内存空间
    uint64_t opParamAddr[opParamVec.size()];
    uint64_t opParamSize = sizeof(OpParam);
    for (uint32_t i = 0U; i < opParamVec.size(); ++i) {
        // 申请硬件内存
        std::string tagParam = ctxTag + "_" + std::to_string(i);
        void *opParamPtr = nullptr;
        if (HcclEngineCtxGet(comm, &tagParam, ccTilingList[i]->commEngine, &opParamSize, &opParamPtr) == HCCL_SUCCESS) {
            HCCL_INFO("HcclEngineCtxGet success, tagParam[%s], opParamAddr[%p], opParamSize[%u]", tagParam.c_str(), opParamPtr, opParamSize);
            opParamAddr[i] = reinterpret_cast<uint64_t>(opParamPtr);
        } else {
            CHK_RET(HcclEngineCtxCreate(comm, &tagParam, ccTilingList[i]->commEngine, &opParamSize, &opParamPtr));
            opParamAddr[i] = reinterpret_cast<uint64_t>(opParamPtr);
        }
        HCCL_INFO("HcclAllocOpResCtx the %dth opParam: opParamAddr[%u], opParamSize[%u]", i, opParamAddr[i], opParamSize);

        // 复制数据到硬件内存
        CHK_RET(aclrtMemcpy(reinterpret_cast<void **>(opParamAddr[i]),  opParamSize, &opParamVec[i], opParamSize, aclrtMemcpyKind(1)));
        // 记录OpParam的地址
        resCtx.algInfo[i].opParam = opParamAddr[i];
        // 记录OpParam的偏移量
        resCtx.algInfo[i].offset = initTiling->offset[i];
    }

    // 2. 申请WorkSpace的内存空间
    uint64_t memSize = 20 * 1024 * 1024;
    resCtx.workSpaceSize = memSize;
    // 申请硬件内存
    std::string tagWorkSpace = ctxTag + "_workSpace";
    void *workSpacePtr = nullptr;
    if (HcclEngineCtxGet(comm, &tagWorkSpace, ccTilingList[i]->commEngine, &memSize, &workSpacePtr) == HCCL_SUCCESS) {
        HCCL_INFO("HcclEngineCtxGet success, tagWorkSpace[%s], workSpaceAddr[%p], workSpaceSize[%u]", tagWorkSpace.c_str(), workSpacePtr, memSize);
        resCtx.workSpace = reinterpret_cast<uint64_t>(workSpacePtr);
    } else {
        CHK_RET(HcclEngineCtxCreate(comm, &tagWorkSpace, ccTilingList[i]->commEngine, &memSize, &workSpacePtr));
        resCtx.workSpace = reinterpret_cast<uint64_t>(workSpacePtr);
    }
    HCCL_INFO("HcclAllocOpResCtx the workSpace: workSpaceAddr[%u], workSpaceSize[%u]", resCtx.workSpace, memSize);

    // 3. 获取rankID和ranksize
    CHK_RET(HcclGetRankSize(comm, &resCtx.rankSize));
    CHK_RET(HcclGetRankId(comm, &resCtx.rankId));

    // 4. 申请OpResCtx的内存空间
    std::string tagOpResCtx = ctxTag + "_opResCtx";
    if (HcclEngineCtxGet(comm, &tagOpResCtx, ccTilingList[i]->commEngine, &sizeof(OpResCtx), opResCtxPtr) == HCCL_SUCCESS) {
        HCCL_INFO("HcclEngineCtxGet success, tagOpResCtx[%s], opResCtxAddr[%p], opResCtxSize[%u]", tagOpResCtx.c_str(), opResCtxPtr, sizeof(OpResCtx));
    } else {
        CHK_RET(HcclEngineCtxCreate(comm, &tagOpResCtx, ccTilingList[i]->commEngine, &sizeof(OpResCtx), opResCtxPtr));
    }

    HCCL_INFO("HcclAllocOpResCtx the opResCtx: opResCtxAddr[%u], opResCtxSize[%u]", opResCtxPtr, sizeof(OpResCtx));

       // 5. 复制OpResCtx到硬件内存
    CHK_RET(aclrtMemcpy(opResCtxPtr, sizeof(OpResCtx), &resCtx, sizeof(OpResCtx), aclrtMemcpyKind(1)));

    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAllGather(const Mc2CcTilingInner *ccTiling, OpParam *opParam)
{
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAllReduce(const Mc2CcTilingInner *ccTiling, OpParam *opParam)
{
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForReduceScatter(const Mc2CcTilingInner *ccTiling, OpParam *opParam)
{
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAlltoAll(const Mc2CcTilingInner *ccTiling, OpParam *opParam)
{
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAlltoAllV(const Mc2CcTilingInner *ccTiling, OpParam *opParam)
{
    return HCCL_SUCCESS;
}

typedef HcclResult (*OpParamPrepareFunc)(const Mc2CcTilingInner *ccTiling, OpParam *opParam);

std::unordered_map<HcclCMDType, OpParamPrepareFunc> opParamPrepareFuncMap = {
    {HcclCMDType::HcclAllGather, PrepareParamForAllGather},
    {HcclCMDType::HcclAllReduce, PrepareParamForAllReduce},
    {HcclCMDType::HcclReduceScatter, PrepareParamForReduceScatter},
    {HcclCMDType::HcclAlltoAll, PrepareParamForAlltoAll},
    {HcclCMDType::HcclAlltoAllV, PrepareParamForAlltoAllV},
};

HcclResult PrepareOpParams(const char *topoTag, const Mc2CcTilingInner *ccTiling, OpParam *opParam)
{
    auto it = opParamPrepareFuncMap.find(ccTiling->opType);
    if (it != opParamPrepareFuncMap.end()) {
        return it->second(ccTiling, opParam);
    }
    HCCL_ERROR("PrepareOpParams error, opType[%d] not found", ccTiling->opType);
    return HCCL_E_INTERNAL;
}




HcclResult GetOpParam(const char *topoTag, const Mc2CcTilingInner *ccTiling, OpParam *opParam)
{
    opParam->opType = static_cast<HcclCMDType>(ccTiling->opType);
    CHK_RET(PrepareOpParams(topoTag, ccTiling, opParam));

}

#endif
