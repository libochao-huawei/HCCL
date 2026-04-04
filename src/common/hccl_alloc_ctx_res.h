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
#include "adapter_error_manager_pub.h"
#include "alg_param.h"
#include "op_common.h"
#include "alg_env_config.h"
#include "hccl_res_expt.h"
#include "load_kernel.h"
#include "coll_alg_v2_exec_registry.h"

#include <vector>

using namespace ops_hccl;

// Forward declarations for types that might not be fully defined
namespace ops_hccl {
    struct OpParam;
    struct TopoInfoWithNetLayerDetails;
    struct AlgResourceCtxSerializable;
    class ExecuteSelector;
    class InsCollAlgBase;
    class CollAlgExecRegistryV2;
}

constexpr uint32_t MC2_TILING_VERSION = 2U;
constexpr uint32_t MAX_HCOM_NUM = 3U;

constexpr uint32_t INIT_TILING_VERSION = 100U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;

struct Mc2ServerCfg {
    uint32_t version;
    uint8_t  debugMode;
    uint8_t  sendArgIndex;
    uint8_t  recvArgIndex;
    uint8_t  commOutArgIndex;
    uint8_t  reserved[8];
};

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
    uint32_t rankId;
    uint32_t rankSize;
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

HcclResult CheckCommEngine(const void *ccTilingList[], uint32_t tilingNum)
{
    for (uint32_t i = 0U; i < tilingNum; ++i) {
        const Mc2CcTilingInner *ccTiling = static_cast<const Mc2CcTilingInner *>(ccTilingList[i]);
        if (ccTiling->commEngine != static_cast<uint8_t>(COMM_ENGINE_AICPU)) {
            HCCL_ERROR("Invalid commEngine %u.", ccTiling->commEngine);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAllocOpResCtx(HcclComm comm, const std::string &ctxTag, const std::vector<OpParam> &opParamVec, void* mc2Tiling, const void *ccTilingList[], void** opResCtxPtr)
{
    CHK_PTR_NULL(opResCtxPtr);
    OpResCtx resCtx;
    Mc2InitTilingInner *initTiling = static_cast<Mc2InitTilingInner *>(mc2Tiling);

    // 1. 申请存放OpParam的内存空间
    std::vector<uint64_t> opParamAddr(opParamVec.size());
    uint64_t opParamSize = sizeof(OpParam);
    for (uint32_t i = 0U; i < opParamVec.size(); ++i) {
        // 申请硬件内存
        std::string tagParam = ctxTag + "_" + std::to_string(i);
        void *opParamPtr = nullptr;
        const Mc2CcTilingInner *ccTiling = static_cast<const Mc2CcTilingInner *>(ccTilingList[i]);
        if (HcclEngineCtxGet(comm, tagParam.c_str(), static_cast<CommEngine>(ccTiling->commEngine), &opParamPtr, &opParamSize) == HCCL_SUCCESS) {
            HCCL_INFO("HcclEngineCtxGet success, tagParam[%s], opParamAddr[%p], opParamSize[%u]", tagParam.c_str(), opParamPtr, opParamSize);
            opParamAddr[i] = reinterpret_cast<uint64_t>(opParamPtr);
        } else {
            CHK_RET(HcclEngineCtxCreate(comm, tagParam.c_str(), static_cast<CommEngine>(ccTiling->commEngine), opParamSize, &opParamPtr));
            opParamAddr[i] = reinterpret_cast<uint64_t>(opParamPtr);
        }
        HCCL_INFO("HcclAllocOpResCtx the %dth opParam: opParamAddr[%u], opParamSize[%u]", i, opParamAddr[i], opParamSize);

        // 复制数据到硬件内存
        aclError aclRet = aclrtMemcpy(reinterpret_cast<void *>(opParamAddr[i]), opParamSize, &opParamVec[i], opParamSize, aclrtMemcpyKind(1));
        CHK_RET(aclRet == ACL_ERROR_NONE ? HCCL_SUCCESS : HCCL_E_RUNTIME);
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
    const Mc2CcTilingInner *ccTiling0 = static_cast<const Mc2CcTilingInner *>(ccTilingList[0]);
    if (HcclEngineCtxGet(comm, tagWorkSpace.c_str(), static_cast<CommEngine>(ccTiling0->commEngine), &workSpacePtr, &memSize) == HCCL_SUCCESS) {
        HCCL_INFO("HcclEngineCtxGet success, tagWorkSpace[%s], workSpaceAddr[%p], workSpaceSize[%u]", tagWorkSpace.c_str(), workSpacePtr, memSize);
        resCtx.workSpace = reinterpret_cast<uint64_t>(workSpacePtr);
    } else {
        CHK_RET(HcclEngineCtxCreate(comm, tagWorkSpace.c_str(), static_cast<CommEngine>(ccTiling0->commEngine), memSize, &workSpacePtr));
        resCtx.workSpace = reinterpret_cast<uint64_t>(workSpacePtr);
    }
    HCCL_INFO("HcclAllocOpResCtx the workSpace: workSpaceAddr[%u], workSpaceSize[%u]", resCtx.workSpace, memSize);

    // 3. 获取rankID和ranksize
    CHK_RET(HcclGetRankSize(comm, &resCtx.rankSize));
    CHK_RET(HcclGetRankId(comm, &resCtx.rankId));

    // 4. 申请OpResCtx的内存空间
    std::string tagOpResCtx = ctxTag + "_opResCtx";
    uint64_t opResCtxSize = sizeof(OpResCtx);
    if (HcclEngineCtxGet(comm, tagOpResCtx.c_str(), static_cast<CommEngine>(ccTiling0->commEngine), opResCtxPtr, &opResCtxSize) == HCCL_SUCCESS) {
        HCCL_INFO("HcclEngineCtxGet success, tagOpResCtx[%s], opResCtxAddr[%p], opResCtxSize[%u]", tagOpResCtx.c_str(), opResCtxPtr, opResCtxSize);
    } else {
        CHK_RET(HcclEngineCtxCreate(comm, tagOpResCtx.c_str(), static_cast<CommEngine>(ccTiling0->commEngine), opResCtxSize, opResCtxPtr));
    }

    HCCL_INFO("HcclAllocOpResCtx the opResCtx: opResCtxAddr[%u], opResCtxSize[%u]", opResCtxPtr, opResCtxSize);

    // 5. 复制OpResCtx到硬件内存
    aclError aclRet = aclrtMemcpy(*opResCtxPtr, opResCtxSize, &resCtx, opResCtxSize, aclrtMemcpyKind(1));
    CHK_RET(aclRet == ACL_ERROR_NONE ? HCCL_SUCCESS : HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAllGather(const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &param)
{
    HCCL_INFO("PrepareParamForAllGather, ccTiling[%p]", ccTiling);
  
    param.opMode = OpMode::OPBASE;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    // 参数准备
    param.inputPtr = 0;
    param.inputSize = 0;
    param.outputPtr = 0;
    param.outputSize = 0;
    param.DataDes.count = 0;
    param.DataDes.dataType = static_cast<HcclDataType>(ccTiling->srcDataType);
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    param.enableDetour = false;
    param.deviceType = deviceType;
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAllReduce(const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &param)
{
    HCCL_INFO("PrepareParamForAllReduce, ccTiling[%p]", ccTiling);
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForReduceScatter(const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &param)
{
    HCCL_INFO("PrepareParamForReduceScatter, ccTiling[%p]", ccTiling);
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAlltoAll(const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &param)
{
    HCCL_INFO("PrepareParamForAlltoAll, ccTiling[%p]", ccTiling);
    return HCCL_SUCCESS;
}

HcclResult PrepareParamForAlltoAllV(const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &param)
{
    HCCL_INFO("PrepareParamForAlltoAllV, ccTiling[%p]", ccTiling);
    return HCCL_SUCCESS;
}

typedef HcclResult (*OpParamPrepareFunc)(const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &opParam);

std::unordered_map<HcclCMDType, OpParamPrepareFunc> opParamPrepareFuncMap = {
    {HCCL_CMD_ALLGATHER, PrepareParamForAllGather},
    {HCCL_CMD_ALLREDUCE, PrepareParamForAllReduce},
    {HCCL_CMD_REDUCE_SCATTER, PrepareParamForReduceScatter},
    {HCCL_CMD_ALLTOALL, PrepareParamForAlltoAll},
    {HCCL_CMD_ALLTOALLV, PrepareParamForAlltoAllV},
};

HcclResult PrepareOpParams(const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &opParam)
{
    auto it = opParamPrepareFuncMap.find(static_cast<HcclCMDType>(ccTiling->opType));
    if (it != opParamPrepareFuncMap.end()) {
        return it->second(tag, ccTiling, opParam);
    }
    HCCL_ERROR("PrepareOpParams error, opType[%d] not found", ccTiling->opType);
    return HCCL_E_INTERNAL;
}

HcclResult InitOpParamByTiling(HcclComm comm, void *stream, const std::string &tag,
    const Mc2CcTilingInner *ccTiling, OpParam &opParam)
{
    opParam.opType = static_cast<HcclCMDType>(ccTiling->opType);
    opParam.stream = reinterpret_cast<aclrtStream>(stream);
    CHK_RET(HcclGetCommName(comm, opParam.commName));
    CHK_RET(PrepareOpParams(tag, ccTiling, opParam));
    return HCCL_SUCCESS;
}

HcclResult SelectAlgAndPrepareEngine(HcclComm comm, OpParam &opParam, std::string &algName,
    std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo)
{
    opParam.hcclComm = comm;
    CHK_RET(HcclGetOpExpansionMode(comm, opParam));
    CHK_RET(HcclCalcTopoInfo(comm, opParam, topoInfo));
    CHK_RET(CheckAsymmetricTopoSupport(opParam.opType, topoInfo.get()));

    std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
    CHK_RET(collAlgSelector->Run(opParam, topoInfo.get(), algName));
    if (algName.empty()) {
        HCCL_ERROR("[Selector] select algname fail!");
        return HCCL_E_PTR;
    }

    CHK_RET(SetCommEngine(opParam));
    if (GetExternalInputHcclAivOnlyMode() && opParam.engine != COMM_ENGINE_AIV) {
        HCCL_ERROR("[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(opParam.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    if ((opParam.engine == COMM_ENGINE_AICPU_TS) || (opParam.engine == COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[Selector] is aicpu mode");
        CHK_RET(LoadAICPUKernel());
    }
    CHK_RET(SetOpParamAlgTag(opParam, algName));
    return HCCL_SUCCESS;
}

HcclResult HandleSingleRankAndCommMode(HcclComm comm, OpParam &opParam, bool &skipGetRes)
{
    uint32_t userRankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(opParam));
        skipGetRes = true;
        return HCCL_SUCCESS;
    }

    bool isOpBase = true;
    const char* opModeStr = isOpBase ? "_opbase" : "_offload";
    auto ret = sprintf_s(opParam.commModeTag, sizeof(opParam.commModeTag), "%s_%s", opParam.commName, opModeStr);
    if (ret <= 0) {
        HCCL_ERROR("[%s] failed to fill opParam.commModeTag", __func__);
        return HCCL_E_INTERNAL;
    }
    skipGetRes = false;
    return HCCL_SUCCESS;
}

HcclResult GetOpParamResCtx(HcclComm comm, const std::string &algName, OpParam &opParam,
    TopoInfoWithNetLayerDetails *topoInfo, void **resCtxOut)
{
    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(opParam.opType, algName);
    CHK_PRT_RET(executor.get() == nullptr,
        HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    bool isResourceReused = false;

    ThreadHandle cpuTsThread{0};
    ThreadHandle exportedAicpuTsThread{0};
    if ((opParam.engine == COMM_ENGINE_AICPU_TS) || (opParam.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, opParam.stream, 1, &cpuTsThread));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    }
    CHK_RET(HcclGetAlgRes(comm, opParam, executor, topoInfo, resCtxHost, resCtxOut, isResourceReused));
    opParam.resCtx = *resCtxOut;
    return HCCL_SUCCESS;
}

HcclResult GetOpParam(HcclComm comm, void* stream, const std::string &tag, const Mc2CcTilingInner *ccTiling, OpParam &opParam)
{
    CHK_RET(InitOpParamByTiling(comm, stream, tag, ccTiling, opParam));

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(SelectAlgAndPrepareEngine(comm, opParam, algName, topoInfo));

    bool skipGetRes = false;
    CHK_RET(HandleSingleRankAndCommMode(comm, opParam, skipGetRes));
    if (skipGetRes) {
        return HCCL_SUCCESS;
    }

    void *resCtxSequence = nullptr;
    CHK_RET(GetOpParamResCtx(comm, algName, opParam, topoInfo.get(), &resCtxSequence));
    return HCCL_SUCCESS;
}

#endif
