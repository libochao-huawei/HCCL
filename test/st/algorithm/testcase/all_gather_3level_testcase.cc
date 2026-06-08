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

constexpr uint32_t DATATYPE_SIZE_TABLE_ALL_GATHER_3LEVEL[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_ALL_GATHER_3LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunAllGather3LevelA5(const TopoMeta &topoMeta, const u64 &sendCount, const HcclDataType &dataType)
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_ALL_GATHER_3LEVEL[dataType];
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
            u64 sendBufSize = sendCount * dataTypeSize;
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

// P0: #1 - 3-level basic correctness on 128-card topology
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_8x8x2_fp32_basic)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// P0: #3 - outputRepeatStride>0, repeatNum=L2=3 verification
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x4x3_fp32_repeatnum_gt1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// P0: #16 - backward compatibility, 2-level behavior unchanged
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_2level_backward_compat_meshnhr)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// P1: #5 - different topology scale correctness
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x4x2_int32_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto sendCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// // P1: #6 - asymmetric middle layer (Level1)
// TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_8x4x2_int16_asymmetric_mid)
// {
//     TopoMeta topoMeta;
//     GenTopoMeta(topoMeta, 2, 4, 8);
//     auto sendCount = 1000;
//     auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
//     RunAllGather3LevelA5(topoMeta, sendCount, dataType);
// }

// P1: #10 - small-scale large-data loop segmentation
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x4x2_fp32_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto sendCount = 400 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// P1: #11 - small cluster, boundary rank verification, sendCount=200+1
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_8x2x2_fp32_small_cluster_send200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto sendCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// P2: #4 - higher repeatNum (repeatNum=L2=4)
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x2x4_int32_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// // P2: #7 - FP16 data type on 32-card topology
// TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x4x2_fp16_dtype)
// {
//     TopoMeta topoMeta;
//     GenTopoMeta(topoMeta, 2, 4, 4);
//     auto sendCount = 500 * 1024;
//     auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
//     RunAllGather3LevelA5(topoMeta, sendCount, dataType);
// }

// P2: #8 - BFP16 data type on 16-card topology
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x2x2_bfp16_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto sendCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// // P2: #12 - extremely small topology
// TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x2x2_int8_corner)
// {
//     TopoMeta topoMeta;
//     GenTopoMeta(topoMeta, 2, 2, 4);
//     auto sendCount = 100;
//     auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
//     RunAllGather3LevelA5(topoMeta, sendCount, dataType);
// }

// P2: #14 - Level2 has 3 clusters (repeatNum=3)
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_8x2x3_fp32_level2_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// P2: #15 - fully asymmetric dimensions
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x3x2_int32_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// --- Degenerate Level (dimension=1) edge cases ---

// L1=1: single server per pod, 8x1x3=24 ranks, degenerate L1
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_8x1x3_fp32_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// L1=1: degenerate L1 + sendCount=8+1
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x1x2_int8_l1_degenerate_send8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    auto sendCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// L0=1 + L1=1: double degenerate, 1x1x4=4 ranks, sendCount=16+1
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_1x1x4_fp32_double_degenerate_send16_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    auto sendCount = 16 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// --- Strange / weird sendCount = aligned_value + 1 cases ---

// sendCount=4+1=5: just over power-of-2, tests remainder element in stride slicing
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x3x2_fp32_send4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto sendCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// sendCount=64K+1=65537: just over 64K boundary, loop slicing remainder on 32-card
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x4x2_int16_send64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto sendCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// L0=1 + sendCount=128K+1: degenerate L0 + large data with remainder element
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_1x4x2_fp32_l0_degenerate_send128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto sendCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}