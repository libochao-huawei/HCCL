/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hccl_mc2.h"
#include "hccl/dtype_common.h"

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

// Declarations from hccl_alloc_ctx_res.h.
HcclResult CheckInputParam(const HcclComm comm, const void *mc2Tiling, const aclrtStream stream);
HcclResult HcclGetTilingList(const void *mc2Tiling, const void *p[], uint32_t &cnt);
HcclResult CheckCommEngine(const void *ccTilingList[], uint32_t tilingNum);
HcclResult CheckIsReduce(const Mc2CcTilingInner *ccTiling, bool *isReduce);

extern "C" void SetMc2UtDeviceType(DevType type);

namespace {
constexpr uint32_t MC2_TILING_VERSION = 2U;
constexpr uint32_t MAX_HCOM_NUM = 3U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;

std::vector<uint8_t> BuildMc2Tiling(
    uint32_t version = MC2_TILING_VERSION,
    uint32_t hcomCnt = 1U,
    uint8_t commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU),
    uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE))
{
    constexpr size_t headU32Num = 2U + MAX_CC_TILING_NUM;
    const size_t ccOffset = headU32Num * sizeof(uint32_t);
    std::vector<uint8_t> data(ccOffset + sizeof(Mc2CcTilingInner), 0U);

    auto *head = reinterpret_cast<uint32_t *>(data.data());
    head[0] = version;
    head[1] = hcomCnt;
    head[2] = static_cast<uint32_t>(ccOffset);

    auto *ccTiling = reinterpret_cast<Mc2CcTilingInner *>(data.data() + ccOffset);
    ccTiling->commEngine = commEngine;
    ccTiling->srcDataType = static_cast<uint8_t>(HCCL_DATA_TYPE_FP16);
    ccTiling->dstDataType = static_cast<uint8_t>(HCCL_DATA_TYPE_FP16);
    ccTiling->opType = opType;
    ccTiling->reduceType = static_cast<uint32_t>(HCCL_REDUCE_SUM);
    std::strncpy(ccTiling->groupName, "ut_group", GROUP_NAME_SIZE - 1U);
    std::strncpy(ccTiling->algConfig, "ut_alg", ALG_CONFIG_SIZE - 1U);

    return data;
}
}

class HcclMc2Test : public testing::Test {
protected:
    void SetUp() override
    {
        SetMc2UtDeviceType(DevType::DEV_TYPE_950);
    }
};

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_DeviceNotSupport)
{
    SetMc2UtDeviceType(DevType::DEV_TYPE_910B);
    auto mc2Tiling = BuildMc2Tiling();
    int stream = 0;
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(
        reinterpret_cast<HcclComm>(0x1), &stream, mc2Tiling.data(), &opResCtx);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_NullComm)
{
    auto mc2Tiling = BuildMc2Tiling();
    int stream = 0;
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(nullptr, &stream, mc2Tiling.data(), &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_NullStream)
{
    auto mc2Tiling = BuildMc2Tiling();
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(
        reinterpret_cast<HcclComm>(0x1), nullptr, mc2Tiling.data(), &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_NullMc2Tiling)
{
    int stream = 0;
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(
        reinterpret_cast<HcclComm>(0x1), &stream, nullptr, &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_NullOpResCtx)
{
    auto mc2Tiling = BuildMc2Tiling();
    int stream = 0;

    HcclResult ret = HcclAllocComResourceByTiling(
        reinterpret_cast<HcclComm>(0x1), &stream, mc2Tiling.data(), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_InvalidTilingVersion)
{
    auto mc2Tiling = BuildMc2Tiling(1U, 1U);
    int stream = 0;
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(
        reinterpret_cast<HcclComm>(0x1), &stream, mc2Tiling.data(), &opResCtx);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_InvalidCommEngine)
{
    auto mc2Tiling = BuildMc2Tiling(MC2_TILING_VERSION, 1U, static_cast<uint8_t>(CommEngine::COMM_ENGINE_AIV));
    int stream = 0;
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(
        reinterpret_cast<HcclComm>(0x1), &stream, mc2Tiling.data(), &opResCtx);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclMc2Test, HcclAllocComResourceByTiling_Success)
{
    auto mc2Tiling = BuildMc2Tiling();
    int stream = 0;
    void *opResCtx = nullptr;

    HcclResult ret = HcclAllocComResourceByTiling(
        reinterpret_cast<HcclComm>(0x1), &stream, mc2Tiling.data(), &opResCtx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(opResCtx, nullptr);

    if (opResCtx != nullptr) {
        std::free(opResCtx);
    }
}

TEST_F(HcclMc2Test, HcclGetTilingList_InvalidVersion)
{
    auto mc2Tiling = BuildMc2Tiling(1U, 1U);
    const void *ccTilingList[MAX_CC_TILING_NUM] = {};
    uint32_t tilingNum = 0;

    HcclResult ret = HcclGetTilingList(mc2Tiling.data(), ccTilingList, tilingNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclMc2Test, HcclGetTilingList_InvalidCount)
{
    auto mc2Tiling = BuildMc2Tiling(MC2_TILING_VERSION, MAX_HCOM_NUM + 1U);
    const void *ccTilingList[MAX_CC_TILING_NUM] = {};
    uint32_t tilingNum = 0;

    HcclResult ret = HcclGetTilingList(mc2Tiling.data(), ccTilingList, tilingNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclMc2Test, HcclGetTilingList_ValidInput)
{
    auto mc2Tiling = BuildMc2Tiling();
    const void *ccTilingList[MAX_CC_TILING_NUM] = {};
    uint32_t tilingNum = 0;

    HcclResult ret = HcclGetTilingList(mc2Tiling.data(), ccTilingList, tilingNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tilingNum, 1U);
    EXPECT_EQ(ccTilingList[0], mc2Tiling.data() + (2U + MAX_CC_TILING_NUM) * sizeof(uint32_t));
}

TEST_F(HcclMc2Test, CheckCommEngine_ValidAndInvalid)
{
    Mc2CcTilingInner valid = {};
    valid.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);

    Mc2CcTilingInner invalid = {};
    invalid.commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AIV);

    const void *validList[] = {&valid};
    const void *invalidList[] = {&invalid};

    EXPECT_EQ(CheckCommEngine(validList, 1U), HCCL_SUCCESS);
    EXPECT_EQ(CheckCommEngine(invalidList, 1U), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclMc2Test, CheckInputParam_Coverage)
{
    auto mc2Tiling = BuildMc2Tiling();
    auto comm = reinterpret_cast<HcclComm>(0x1);
    auto stream = reinterpret_cast<aclrtStream>(0x1);

    EXPECT_EQ(CheckInputParam(nullptr, mc2Tiling.data(), stream), HCCL_E_PTR);
    EXPECT_EQ(CheckInputParam(comm, nullptr, stream), HCCL_E_PTR);
    EXPECT_EQ(CheckInputParam(comm, mc2Tiling.data(), nullptr), HCCL_E_PTR);
    EXPECT_EQ(CheckInputParam(comm, mc2Tiling.data(), stream), HCCL_SUCCESS);
}

TEST_F(HcclMc2Test, CheckIsReduce_Coverage)
{
    Mc2CcTilingInner allReduce = {};
    allReduce.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);

    Mc2CcTilingInner allGather = {};
    allGather.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);

    bool isReduce = false;
    EXPECT_EQ(CheckIsReduce(&allReduce, &isReduce), HCCL_SUCCESS);
    EXPECT_TRUE(isReduce);

    EXPECT_EQ(CheckIsReduce(&allGather, &isReduce), HCCL_SUCCESS);
    EXPECT_FALSE(isReduce);
}

TEST_F(HcclMc2Test, KfcOpArgs_BasicCoverage)
{
    void *opArgs = nullptr;
    ASSERT_EQ(HcclKfcAllocOpArgs(&opArgs), HCCL_SUCCESS);
    ASSERT_NE(opArgs, nullptr);

    EXPECT_EQ(HcclKfcOpArgsSetSrcDataType(opArgs, static_cast<uint8_t>(HCCL_DATA_TYPE_FP16)), HCCL_SUCCESS);
    EXPECT_EQ(HcclKfcOpArgsSetDstDataType(opArgs, static_cast<uint8_t>(HCCL_DATA_TYPE_FP16)), HCCL_SUCCESS);
    EXPECT_EQ(HcclKfcOpArgsSetReduceType(opArgs, static_cast<uint32_t>(HCCL_REDUCE_SUM)), HCCL_SUCCESS);
    EXPECT_EQ(HcclKfcOpArgsSetCount(opArgs, 1024U), HCCL_SUCCESS);
    EXPECT_EQ(HcclKfcOpArgsSetCount(opArgs, UINT64_MAX), HCCL_E_PARA);
    EXPECT_EQ(HcclKfcOpArgsSetCommEngine(opArgs, static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU)), HCCL_SUCCESS);
    EXPECT_EQ(HcclKfcOpArgsSetCommEngine(opArgs, 0xFF), HCCL_E_NOT_SUPPORT);

    EXPECT_EQ(HcclKfcFreeOpArgs(opArgs), HCCL_SUCCESS);
    EXPECT_EQ(HcclKfcFreeOpArgs(nullptr), HCCL_E_PTR);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
