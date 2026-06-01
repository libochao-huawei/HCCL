/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

constexpr u32 DATATYPE_SIZE_TABLE_RSV[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_REDUCESCATTERV_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

static inline u32 AnalyseRankSizeRSV(const TopoMeta &topoInfo)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoInfo) {
        for (const auto &podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}

void RunReduceScatterVMultilevel(const TopoMeta &topoInfo, VDataDesTag vDataDes, HcclReduceOp reduceOp)
{
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    auto rankSize = AnalyseRankSizeRSV(topoInfo);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_RSV[vDataDes.dataType];

    u64 sendDataCount = 0;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        sendDataCount += vDataDes.counts[rankId];
    }

    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 recvDataCount = vDataDes.counts[rankId];
            u64 sendBufSize = sendDataCount * dataTypeSize;
            u64 recvBufSize = recvDataCount * dataTypeSize;
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            CHK_RET(HcclReduceScatterV(sendBuf, vDataDes.counts.data(), vDataDes.displs.data(),
                recvBuf, recvDataCount, vDataDes.dataType, reduceOp, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduceScatterV(taskQueues, rankSize, reduceOp, vDataDes);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_aicpu_test)
{
    TopoMeta topoMeta{{{0, 1}}};

    VDataDesTag vDataDes;
    vDataDes.counts = {155, 155};
    vDataDes.displs = {0, 155};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT16;

    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_multilevel_2pod_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};

    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_multilevel_2pod_6rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}}};

    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}