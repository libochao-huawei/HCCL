/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl_mc2.h"
#include "acl/acl_rt.h"
#include "alg_env_config.h"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using namespace HcclSim;
using namespace ops_hccl;

namespace {
constexpr uint32_t MC2_TILING_VERSION = 2U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;

struct Mc2TilingHeader {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
};

struct Mc2CcTilingInnerLite {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};

void CopyLiteral(char *dst, size_t dstSize, const char *src)
{
    if (dst == nullptr || src == nullptr || dstSize == 0U) {
        return;
    }
    size_t i = 0U;
    while (i + 1U < dstSize && src[i] != '\0') {
        dst[i] = src[i];
        ++i;
    }
    dst[i] = '\0';
}

void CopyBytes(uint8_t *dst, size_t dstSize, const uint8_t *src, size_t copySize)
{
    if (dst == nullptr || src == nullptr || dstSize < copySize) {
        return;
    }
    for (size_t i = 0U; i < copySize; ++i) {
        dst[i] = src[i];
    }
}

std::vector<uint8_t> BuildMc2TilingData()
{
    Mc2TilingHeader header {};
    header.version = MC2_TILING_VERSION;
    header.mc2HcommCnt = 1U;
    header.offset[0] = sizeof(Mc2TilingHeader);

    Mc2CcTilingInnerLite ccTiling {};
    ccTiling.version = 2U;
    ccTiling.commEngine = static_cast<uint8_t>(COMM_ENGINE_AICPU);
    ccTiling.srcDataType = static_cast<uint8_t>(HCCL_DATA_TYPE_FP16);
    ccTiling.dstDataType = static_cast<uint8_t>(HCCL_DATA_TYPE_FP16);
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.reduceType = static_cast<uint32_t>(HCCL_REDUCE_SUM);
    CopyLiteral(ccTiling.groupName, sizeof(ccTiling.groupName), "st_mc2_group");
    CopyLiteral(ccTiling.algConfig, sizeof(ccTiling.algConfig), "st_mc2_alg");

    std::vector<uint8_t> tiling(sizeof(Mc2TilingHeader) + sizeof(Mc2CcTilingInnerLite), 0U);
    CopyBytes(tiling.data(), tiling.size(), reinterpret_cast<const uint8_t *>(&header), sizeof(Mc2TilingHeader));
    CopyBytes(tiling.data() + sizeof(Mc2TilingHeader), sizeof(Mc2CcTilingInnerLite),
        reinterpret_cast<const uint8_t *>(&ccTiling), sizeof(Mc2CcTilingInnerLite));
    return tiling;
}

HcclResult RunAllocComResourceByTilingOnRank(uint32_t rankId, const std::vector<uint8_t> &mc2Tiling)
{
    if (aclrtSetDevice(rankId) != ACL_SUCCESS) {
        return HCCL_E_RUNTIME;
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        return HCCL_E_RUNTIME;
    }

    HcclComm comm = nullptr;
    HcclResult ret = HcclCommInitClusterInfo("./ranktable.json", rankId, &comm);
    if (ret != HCCL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        return ret;
    }

    void *opResCtx = nullptr;
    ret = HcclAllocComResourceByTiling(comm, stream, const_cast<uint8_t *>(mc2Tiling.data()), &opResCtx);
    if (ret == HCCL_SUCCESS && opResCtx == nullptr) {
        ret = HCCL_E_PTR;
    }

    (void)HcclCommDestroy(comm);
    (void)aclrtDestroyStream(stream);
    return ret;
}
} // namespace

class ST_MC2_ALLOC_RESOURCE_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        setenv("HCCL_ENABLE_OPEN_AICPU", "1", 1);
    }

    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
};

TEST_F(ST_MC2_ALLOC_RESOURCE_TEST, st_mc2_alloc_com_resource_allgather_2rank_success)
{
    TopoMeta topoMeta {{{0, 1}}};
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    const std::vector<uint8_t> mc2Tiling = BuildMc2TilingData();
    constexpr uint32_t rankSize = 2U;
    std::array<HcclResult, rankSize> rets {};
    std::vector<std::thread> threads;
    threads.reserve(rankSize);
    for (uint32_t rankId = 0U; rankId < rankSize; ++rankId) {
        threads.emplace_back([rankId, &mc2Tiling, &rets]() {
            rets[rankId] = RunAllocComResourceByTilingOnRank(rankId, mc2Tiling);
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    for (uint32_t rankId = 0U; rankId < rankSize; ++rankId) {
        EXPECT_EQ(rets[rankId], HCCL_SUCCESS) << "rankId=" << rankId;
    }

    SimWorld::Global()->Deinit();
}
