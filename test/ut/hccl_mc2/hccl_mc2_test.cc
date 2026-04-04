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
#include <array>
#include <vector>

#include "hccl_mc2.h"
#include "hccl/dtype_common.h"

struct Mc2CcTilingInner;

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
constexpr size_t CC_TILING_BYTES = 284U;
constexpr size_t CC_TILING_COMM_ENGINE_OFFSET = 14U;
constexpr size_t CC_TILING_SRC_DATA_TYPE_OFFSET = 15U;
constexpr size_t CC_TILING_DST_DATA_TYPE_OFFSET = 16U;
constexpr size_t CC_TILING_GROUP_NAME_OFFSET = 17U;
constexpr size_t CC_TILING_ALG_CONFIG_OFFSET = 145U;
constexpr size_t CC_TILING_OP_TYPE_OFFSET = 276U;
constexpr size_t CC_TILING_REDUCE_TYPE_OFFSET = 280U;
constexpr size_t CC_TILING_STR_LEN = 128U;

void SetU8(std::vector<uint8_t> &buf, size_t offset, uint8_t value)
{
    buf[offset] = value;
}

void SetU32(std::vector<uint8_t> &buf, size_t offset, uint32_t value)
{
    std::memcpy(buf.data() + offset, &value, sizeof(value));
}

template <size_t N>
void SetLiteral(std::vector<uint8_t> &buf, size_t offset, size_t fieldSize, const char (&text)[N])
{
    const size_t copyLen = (N < fieldSize) ? N : fieldSize;
    std::memcpy(buf.data() + offset, text, copyLen);
    buf[offset + fieldSize - 1U] = '\0';
}

std::vector<uint8_t> BuildMc2Tiling(
    uint32_t version = MC2_TILING_VERSION,
    uint32_t hcomCnt = 1U,
    uint8_t commEngine = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU),
    uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE))
{
    constexpr size_t headU32Num = 2U + MAX_CC_TILING_NUM;
    const size_t ccOffset = headU32Num * sizeof(uint32_t);
    std::vector<uint8_t> data(ccOffset + CC_TILING_BYTES, 0U);

    auto *head = reinterpret_cast<uint32_t *>(data.data());
    head[0] = version;
    head[1] = hcomCnt;
    head[2] = static_cast<uint32_t>(ccOffset);

    SetU8(data, ccOffset + CC_TILING_COMM_ENGINE_OFFSET, commEngine);
    SetU8(data, ccOffset + CC_TILING_SRC_DATA_TYPE_OFFSET, static_cast<uint8_t>(HCCL_DATA_TYPE_FP16));
    SetU8(data, ccOffset + CC_TILING_DST_DATA_TYPE_OFFSET, static_cast<uint8_t>(HCCL_DATA_TYPE_FP16));
    SetU32(data, ccOffset + CC_TILING_OP_TYPE_OFFSET, opType);
    SetU32(data, ccOffset + CC_TILING_REDUCE_TYPE_OFFSET, static_cast<uint32_t>(HCCL_REDUCE_SUM));
    SetLiteral(data, ccOffset + CC_TILING_GROUP_NAME_OFFSET, CC_TILING_STR_LEN, "ut_group");
    SetLiteral(data, ccOffset + CC_TILING_ALG_CONFIG_OFFSET, CC_TILING_STR_LEN, "ut_alg");

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
    std::array<uint8_t, CC_TILING_BYTES> valid = {};
    std::array<uint8_t, CC_TILING_BYTES> invalid = {};
    valid[CC_TILING_COMM_ENGINE_OFFSET] = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AICPU);
    invalid[CC_TILING_COMM_ENGINE_OFFSET] = static_cast<uint8_t>(CommEngine::COMM_ENGINE_AIV);

    const void *validList[] = {valid.data()};
    const void *invalidList[] = {invalid.data()};

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
    std::array<uint8_t, CC_TILING_BYTES> allReduce = {};
    std::array<uint8_t, CC_TILING_BYTES> allGather = {};
    uint32_t reduceOpType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    uint32_t gatherOpType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    std::memcpy(allReduce.data() + CC_TILING_OP_TYPE_OFFSET, &reduceOpType, sizeof(reduceOpType));
    std::memcpy(allGather.data() + CC_TILING_OP_TYPE_OFFSET, &gatherOpType, sizeof(gatherOpType));

    bool isReduce = false;
    EXPECT_EQ(CheckIsReduce(reinterpret_cast<const Mc2CcTilingInner *>(allReduce.data()), &isReduce), HCCL_SUCCESS);
    EXPECT_TRUE(isReduce);

    EXPECT_EQ(CheckIsReduce(reinterpret_cast<const Mc2CcTilingInner *>(allGather.data()), &isReduce), HCCL_SUCCESS);
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
