/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstring>
#include <vector>
#include "hccl_mc2.h"
#include "hccl_alloc_ctx_res.h"
#include "hccl_inner.h"

using namespace testing;
using namespace ops_hccl;

class HcclAllocComResourceByTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
    }

    static void TearDownTestCase()
    {
    }

    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
    }

    void CreateValidMc2Tiling(void **mc2Tiling)
    {
        Mc2InitTilingInner *initTiling = new Mc2InitTilingInner();
        initTiling->version = INIT_TILING_VERSION;
        initTiling->mc2HcommCnt = 1;
        initTiling->offset[0] = 0;
        initTiling->debugMode = 0;
        initTiling->preparePosition = 0;
        initTiling->queueNum = 1;
        initTiling->commBlockNum = 1;
        initTiling->devType = 0;
        memset(initTiling->reserved, 0, sizeof(initTiling->reserved));

        Mc2CcTilingInner *ccTiling = new Mc2CcTilingInner();
        ccTiling->skipLocalRankCopy = 0;
        ccTiling->skipBufferWindowCopy = 0;
        ccTiling->stepSize = 1;
        ccTiling->version = 1;
        memset(ccTiling->reserved, 0, sizeof(ccTiling->reserved));
        ccTiling->commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);
        ccTiling->srcDataType = HCCL_DATA_TYPE_FP16;
        ccTiling->dstDataType = HCCL_DATA_TYPE_FP16;
        strncpy(ccTiling->groupName, "test_group", GROUP_NAME_SIZE - 1);
        strncpy(ccTiling->algConfig, "test_alg", ALG_CONFIG_SIZE - 1);
        ccTiling->opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
        ccTiling->reduceType = static_cast<uint32_t>(HCCL_REDUCE_SUM);

        *mc2Tiling = initTiling;
    }

    void CreateInvalidMc2Tiling(void **mc2Tiling)
    {
        Mc2InitTilingInner *initTiling = new Mc2InitTilingInner();
        initTiling->version = 0;
        initTiling->mc2HcommCnt = 1;
        *mc2Tiling = initTiling;
    }
};

// 测试HcclAllocComResourceByTiling接口，验证当comm为nullptr时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocComResourceByTiling_NullComm)
{
    void *mc2Tiling = nullptr;
    void *stream = nullptr;
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(nullptr, stream, mc2Tiling, &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试HcclAllocComResourceByTiling接口，验证当stream为nullptr时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocComResourceByTiling_NullStream)
{
    void *mc2Tiling = nullptr;
    void *stream = nullptr;
    void *opResCtx = nullptr;
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);

    HcclResult ret = HcclAllocComResourceByTiling(comm, nullptr, mc2Tiling, &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试HcclAllocComResourceByTiling接口，验证当mc2Tiling为nullptr时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocComResourceByTiling_NullMc2Tiling)
{
    void *mc2Tiling = nullptr;
    void *stream = reinterpret_cast<void *>(0x1);
    void *opResCtx = nullptr;
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);

    HcclResult ret = HcclAllocComResourceByTiling(comm, stream, nullptr, &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试HcclAllocComResourceByTiling接口接口，验证当opResCtx为nullptr时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocComResourceByTiling_NullOpResCtx)
{
    void *mc2Tiling = nullptr;
    void *stream = reinterpret_cast<void *>(0x1);
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);

    HcclResult ret = HcclAllocComResourceByTiling(comm, stream, mc2Tiling, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试HcclAllocComResourceByTiling接口，验证当tiling版本无效时返回HCCL_E_PARA
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocComResourceByTiling_InvalidTilingVersion)
{
    void *mc2Tiling = nullptr;
    CreateInvalidMc2Tiling(&mc2Tiling);
    void *stream = reinterpret_cast<void *>(0x1);
    void *opResCtx = nullptr;
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);

    HcclResult ret = HcclAllocComResourceByTiling(comm, stream, mc2Tiling, &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PARA);

    delete static_cast<Mc2InitTilingInner *>(mc2Tiling);
}

// 测试HcclGetTilingList接口，验证当mc2Tiling为nullptr时返回错误
TEST_F(HcclAllocComResourceByTilingTest, HcclGetTilingList_NullMc2Tiling)
{
    const void *ccTilingList[MAX_CC_TILING_NUM];
    uint32_t tilingNum = 0;

    HcclResult ret = HcclGetTilingList(nullptr, ccTilingList, tilingNum);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 测试HcclGetTilingList接口，验证当ccTilingList为nullptr时返回错误
TEST_F(HcclAllocComResourceByTilingTest, HcclGetTilingList_NullCcTilingList)
{
    Mc2InitTilingInner initTiling;
    initTiling.version = INIT_TILING_VERSION;
    initTiling.mc2HcommCnt = 1;
    uint32_t tilingNum = 0;

    HcclResult ret = HcclGetTilingList(&initTiling, nullptr, tilingNum);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 测试HcclGetTilingList接口，验证正常输入时能正确解析tiling数据
TEST_F(HcclAllocComResourceByTilingTest, HcclGetTilingList_ValidInput)
{
    Mc2InitTilingInner initTiling;
    initTiling.version = INIT_TILING_VERSION;
    initTiling.mc2HcommCnt = 1;
    initTiling.offset[0] = sizeof(Mc2InitTilingInner);

    Mc2CcTilingInner ccTiling;
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    ccTiling.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);

    const void *ccTilingList[MAX_CC_TILING_NUM];
    uint32_t tilingNum = 0;

    HcclResult ret = HcclGetTilingList(&initTiling, ccTilingList, tilingNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tilingNum, 1U);
}

// 测试CheckIsReduce接口，验证ALLREDUCE操作被正确识别为reduce操作
TEST_F(HcclAllocComResourceByTilingTest, CheckIsReduce_AllReduce)
{
    Mc2CcTilingInner ccTiling;
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    bool isReduce = false;

    HcclResult ret = CheckIsReduce(&ccTiling, &isReduce);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isReduce);
}

// 测试CheckIsReduce接口，验证REDUCE_SCATTER操作被正确识别为reduce操作
TEST_F(HcclAllocComResourceByTilingTest, CheckIsReduce_ReduceScatter)
{
    Mc2CcTilingInner ccTiling;
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    bool isReduce = false;

    HcclResult ret = CheckIsReduce(&ccTiling, &isReduce);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isReduce);
}

// 测试CheckIsReduce接口，验证ALLGATHER操作被正确识别为非reduce操作
TEST_F(HcclAllocComResourceByTilingTest, CheckIsReduce_AllGather)
{
    Mc2CcTilingInner ccTiling;
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    bool isReduce = false;

    HcclResult ret = CheckIsReduce(&ccTiling, &isReduce);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(isReduce);
}

// 测试CheckCommEngine接口，验证AICPU引擎类型被正确接受
TEST_F(HcclAllocComResourceByTilingTest, CheckCommEngine_ValidCommEngine)
{
    Mc2CcTilingInner ccTiling1;
    ccTiling1.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);

    Mc2CcTilingInner ccTiling2;
    ccTiling2.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);

    const void *ccTilingList[] = {&ccTiling1, &ccTiling2};
    uint32_t tilingNum = 2;

    HcclResult ret = CheckCommEngine(ccTilingList, tilingNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试CheckCommEngine接口，验证AIV引擎类型返回不支持错误
TEST_F(HcclAllocComResourceByTilingTest, CheckCommEngine_InvalidCommEngine)
{
    Mc2CcTilingInner ccTiling;
    ccTiling.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AIV);

    const void *ccTilingList[] = {&ccTiling};
    uint32_t tilingNum = 1;

    HcclResult ret = CheckCommEngine(ccTilingList, tilingNum);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

// 测试CheckInputParam接口，验证当comm为nullptr时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, CheckInputParam_NullComm)
{
    HcclResult ret = CheckInputParam(nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试CheckInputParam接口，验证当mc2Tiling为nullptr时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, CheckInputParam_NullMc2Tiling)
{
    void *stream = reinterpret_cast<void *>(0x1);
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);

    HcclResult ret = CheckInputParam(comm, nullptr, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试CheckInputParam接口，验证当stream为nullptr时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, CheckInputParam_NullStream)
{
    void *mc2Tiling = reinterpret_cast<void *>(0x1);
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);

    HcclResult ret = CheckInputParam(comm, mc2Tiling, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试CheckInputParam接口，验证正常输入时返回HCCL_SUCCESS
TEST_F(HcclAllocComResourceByTilingTest, CheckInputParam_ValidInput)
{
    void *mc2Tiling = reinterpret_cast<void *>(0x1);
    void *stream = reinterpret_cast<void *>(0x1);
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);

    HcclResult ret = CheckInputParam(comm, mc2Tiling, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试HcclAllocOpResCtx接口，验证分配的资源上下文中的workspace、rankId、rankSize等值正确
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocOpResCtx_VerifyOpResCtxValues)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);
    std::string ctxTag = "test_ctx_tag";
    
    std::vector<OpParam> opParamVec(1);
    opParamVec[0].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    opParamVec[0].srcDataType = HCCL_DATA_TYPE_FP16;
    opParamVec[0].dstDataType = HCCL_DATA_TYPE_FP16;
    opParamVec[0].reduceType = HCCL_REDUCE_SUM;
    opParamVec[0].count = 1024;
    opParamVec[0].engine = CommEngine::COMM_ENGINE_AICPU;
    
    Mc2InitTilingInner initTiling;
    initTiling.version = INIT_TILING_VERSION;
    initTiling.mc2HcommCnt = 1;
    initTiling.offset[0] = 0;
    
    Mc2CcTilingInner ccTiling;
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    ccTiling.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);
    
    const void *ccTilingList[] = {&ccTiling};
    void *opResCtxPtr = nullptr;
    
    HcclResult ret = HcclAllocOpResCtx(comm, ctxTag, opParamVec, &initTiling, ccTilingList, &opResCtxPtr);
    
    if (ret == HCCL_SUCCESS && opResCtxPtr != nullptr) {
        OpResCtx *resCtx = static_cast<OpResCtx *>(opResCtxPtr);
        
        EXPECT_NE(resCtx->workSpace, 0ULL);
        EXPECT_EQ(resCtx->workSpaceSize, 20ULL * 1024 * 1024);
        
        EXPECT_GE(resCtx->rankId, 0ULL);
        EXPECT_GE(resCtx->rankSize, 1ULL);
        
        EXPECT_NE(resCtx->algInfo[0].opParam, 0ULL);
        EXPECT_EQ(resCtx->algInfo[0].offset, 0ULL);
    }
}

// 测试HcclAllocOpResCtx接口，验证多个OpParam的offset值正确设置
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocOpResCtx_VerifyOpParamValues)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);
    std::string ctxTag = "test_ctx_tag_opparam";
    
    std::vector<OpParam> opParamVec(2);
    
    opParamVec[0].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    opParamVec[0].srcDataType = HCCL_DATA_TYPE_FP16;
    opParamVec[0].dstDataType = HCCL_DATA_TYPE_FP16;
    opParamVec[0].reduceType = HCCL_REDUCE_SUM;
    opParamVec[0].count = 1024;
    opParamVec[0].engine = CommEngine::COMM_ENGINE_AICPU;
    
    opParamVec[1].opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    opParamVec[1].srcDataType = HCCL_DATA_TYPE_FP32;
    opParamVec[1].dstDataType = HCCL_DATA_TYPE_FP32;
    opParamVec[1].reduceType = HCCL_REDUCE_SUM;
    opParamVec[1].count = 2048;
    opParamVec[1].engine = CommEngine::COMM_ENGINE_AICPU;
    
    Mc2InitTilingInner initTiling;
    initTiling.version = INIT_TILING_VERSION;
    initTiling.mc2HcommCnt = 2;
    initTiling.offset[0] = 0;
    initTiling.offset[1] = 256;
    
    Mc2CcTilingInner ccTiling1;
    ccTiling1.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    ccTiling1.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);
    
    Mc2CcTilingInner ccTiling2;
    ccTiling2.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling2.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);
    
    const void *ccTilingList[] = {&ccTiling1, &ccTiling2};
    void *opResCtxPtr = nullptr;
    
    HcclResult ret = HcclAllocOpResCtx(comm, ctxTag, opParamVec, &initTiling, ccTilingList, &opResCtxPtr);
    
    if (ret == HCCL_SUCCESS && opResCtxPtr != nullptr) {
        OpResCtx *resCtx = static_cast<OpResCtx *>(opResCtxPtr);
        
        EXPECT_NE(resCtx->workSpace, 0ULL);
        EXPECT_EQ(resCtx->workSpaceSize, 20ULL * 1024 * 1024);
        
        EXPECT_NE(resCtx->algInfo[0].opParam, 0ULL);
        EXPECT_EQ(resCtx->algInfo[0].offset, 0ULL);
        
        EXPECT_NE(resCtx->algInfo[1].opParam, 0ULL);
        EXPECT_EQ(resCtx->algInfo[1].offset, 256ULL);
    }
}

// 测试HcclAllocOpResCtx接口，验证多个OpParam场景下的资源分配正确性
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocOpResCtx_VerifyMultipleOpParams)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);
    std::string ctxTag = "test_ctx_tag_multi";
    
    std::vector<OpParam> opParamVec(3);
    
    for (uint32_t i = 0; i < 3; ++i) {
        opParamVec[i].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        opParamVec[i].srcDataType = HCCL_DATA_TYPE_FP16;
        opParamVec[i].dstDataType = HCCL_DATA_TYPE_FP16;
        opParamVec[i].reduceType = HCCL_REDUCE_SUM;
        opParamVec[i].count = 1024 * (i + 1);
        opParamVec[i].engine = CommEngine::COMM_ENGINE_AICPU;
    }
    
    Mc2InitTilingInner initTiling;
    initTiling.version = INIT_TILING_VERSION;
    initTiling.mc2HcommCnt = 3;
    initTiling.offset[0] = 0;
    initTiling.offset[1] = 512;
    initTiling.offset[2] = 1024;
    
    Mc2CcTilingInner ccTilings[3];
    const void *ccTilingList[3];
    
    for (uint32_t i = 0; i < 3; ++i) {
        ccTilings[i].opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
        ccTilings[i].commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);
        ccTilingList[i] = &ccTilings[i];
    }
    
    void *opResCtxPtr = nullptr;
    
    HcclResult ret = HcclAllocOpResCtx(comm, ctxTag, opParamVec, &initTiling, ccTilingList, &opResCtxPtr);
    
    if (ret == HCCL_SUCCESS && opResCtxPtr != nullptr) {
        OpResCtx *resCtx = static_cast<OpResCtx *>(opResCtxPtr);
        
        EXPECT_NE(resCtx->workSpace, 0ULL);
        EXPECT_EQ(resCtx->workSpaceSize, 20ULL * 1024 * 1024);
        
        for (uint32_t i = 0; i < 3; ++i) {
            printf("[HcclAllocOpResCtx_VerifyMultipleOpParams] opParamVec[%d].opType = %d\n", i, opParamVec[i].opType);
            EXPECT_NE(resCtx->algInfo[i].opParam, 0ULL);
            EXPECT_EQ(resCtx->algInfo[i].offset, initTiling.offset[i]);
        }
    }
}

// 测试HcclAllocOpResCtx接口，验证不同数据类型（FP32->INT32）场景下的资源分配正确性
TEST_F(HcclAllocComResourceByTilingTest, HcclAllocOpResCtx_VerifyDataTypeValues)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1);
    std::string ctxTag = "test_ctx_tag_datatype";
    
    std::vector<OpParam> opParamVec(1);
    opParamVec[0].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    opParamVec[0].srcDataType = HCCL_DATA_TYPE_FP32;
    opParamVec[0].dstDataType = HCCL_DATA_TYPE_INT32;
    opParamVec[0].reduceType = HCCL_REDUCE_MAX;
    opParamVec[0].count = 512;
    opParamVec[0].engine = CommEngine::COMM_ENGINE_AICPU;
    
    Mc2InitTilingInner initTiling;
    initTiling.version = INIT_TILING_VERSION;
    initTiling.mc2HcommCnt = 1;
    initTiling.offset[0] = 128;
    
    Mc2CcTilingInner ccTiling;
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    ccTiling.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);
    
    const void *ccTilingList[] = {&ccTiling};
    void *opResCtxPtr = nullptr;
    
    HcclResult ret = HcclAllocOpResCtx(comm, ctxTag, opParamVec, &initTiling, ccTilingList, &opResCtxPtr);
    
    if (ret == HCCL_SUCCESS && opResCtxPtr != nullptr) {
        OpResCtx *resCtx = static_cast<OpResCtx *>(opResCtxPtr);
        
        printf("[HcclAllocOpResCtx_VerifyDataTypeValues] resCtx->workSpaceSize = %llu, resCtx->algInfo[0].opParam = %d, resCtx->algInfo[0].offset = %llu\n", resCtx->workSpaceSize, resCtx->algInfo[0].opParam, resCtx->algInfo[0].offset);
        
        EXPECT_NE(resCtx->workSpaceSize, 0ULL);
        EXPECT_NE(resCtx->algInfo[0].opParam, 0ULL);
        EXPECT_EQ(resCtx->algInfo[0].offset, 128ULL);
    }
}

// 测试HcclKfcAllocOpArgs接口，验证成功分配OpArgs内存
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcAllocOpArgs_Success)
{
    void *opArgs = nullptr;
    
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(opArgs, nullptr);
    
    if (opArgs != nullptr) {
        HcclKfcFreeOpArgs(opArgs);
    }
}

// 测试HcclKfcAllocOpArgs接口，验证空指针输入时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcAllocOpArgs_NullPtr)
{
    HcclResult ret = HcclKfcAllocOpArgs(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试HcclKfcOpArgsSetSrcDataType接口，验证成功设置源数据类型
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcOpArgsSetSrcDataType_Success)
{
    void *opArgs = nullptr;
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    
    ret = HcclKfcOpArgsSetSrcDataType(opArgs, static_cast<uint8_t>(HCCL_DATA_TYPE_FP32));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    
    HcclKfcFreeOpArgs(opArgs);
}

// 测试HcclKfcOpArgsSetDstDataType接口，验证成功设置目标数据类型
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcOpArgsSetDstDataType_Success)
{
    void *opArgs = nullptr;
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    
    ret = HcclKfcOpArgsSetDstDataType(opArgs, static_cast<uint8_t>(HCCL_DATA_TYPE_INT32));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    
    HcclKfcFreeOpArgs(opArgs);
}

// 测试HcclKfcOpArgsSetReduceType接口，验证成功设置reduce类型
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcOpArgsSetReduceType_Success)
{
    void *opArgs = nullptr;
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    
    ret = HcclKfcOpArgsSetReduceType(opArgs, static_cast<uint32_t>(HCCL_REDUCE_MAX));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    
    HcclKfcFreeOpArgs(opArgs);
}

// 测试HcclKfcOpArgsSetCount接口，验证成功设置数据count
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcOpArgsSetCount_Success)
{
    void *opArgs = nullptr;
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    
    ret = HcclKfcOpArgsSetCount(opArgs, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    
    HcclKfcFreeOpArgs(opArgs);
}

// 测试HcclKfcOpArgsSetCount接口，验证设置无效count时返回HCCL_E_PARA
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcOpArgsSetCount_InvalidCount)
{
    void *opArgs = nullptr;
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    
    ret = HcclKfcOpArgsSetCount(opArgs, UINT64_MAX);
    EXPECT_EQ(ret, HCCL_E_PARA);
    
    HcclKfcFreeOpArgs(opArgs);
}

// 测试HcclKfcOpArgsSetCommEngine接口，验证成功设置通信引擎类型
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcOpArgsSetCommEngine_Success)
{
    void *opArgs = nullptr;
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    
    ret = HcclKfcOpArgsSetCommEngine(opArgs, static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    
    HcclKfcFreeOpArgs(opArgs);
}

// 测试HcclKfcOpArgsSetCommEngine接口，验证设置无效引擎类型时返回HCCL_E_NOT_SUPPORT
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcOpArgsSetCommEngine_InvalidEngine)
{
    void *opArgs = nullptr;
    HcclResult ret = HcclKfcAllocOpArgs(&opArgs);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    
    ret = HcclKfcOpArgsSetCommEngine(opArgs, 255);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    
    HcclKfcFreeOpArgs(opArgs);
}

// 测试HcclKfcFreeOpArgs接口，验证空指针输入时返回HCCL_E_PTR
TEST_F(HcclAllocComResourceByTilingTest, HcclKfcFreeOpArgs_NullPtr)
{
    HcclResult ret = HcclKfcFreeOpArgs(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试HcclGetTilingList接口，验证无效版本时返回错误
TEST_F(HcclAllocComResourceByTilingTest, HcclGetTilingList_InvalidVersion)
{
    Mc2InitTilingInner initTiling;
    initTiling.version = 0;
    initTiling.mc2HcommCnt = 1;
    
    const void *ccTilingList[MAX_CC_TILING_NUM];
    uint32_t tilingNum = 0;
    
    HcclResult ret = HcclGetTilingList(&initTiling, ccTilingList, tilingNum);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 测试HcclGetTilingList接口，验证超过最大数量的tiling时返回错误
TEST_F(HcclAllocComResourceByTilingTest, HcclGetTilingList_InvalidCount)
{
    Mc2InitTilingInner initTiling;
    initTiling.version = INIT_TILING_VERSION;
    initTiling.mc2HcommCnt = MAX_CC_TILING_NUM + 1;
    
    const void *ccTilingList[MAX_CC_TILING_NUM];
    uint32_t tilingNum = 0;
    
    HcclResult ret = HcclGetTilingList(&initTiling, ccTilingList, tilingNum);
    EXPECT_NE(ret, HCCL_SUCCESS);
}
