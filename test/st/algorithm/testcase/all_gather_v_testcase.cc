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

class ST_ALL_GATHER_V_TEST : public ::testing::Test {
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

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_aicpu_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {89478485, 178956970};
    vDataDes.displs = {0, 89478485};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

static inline u32 AnalyseRankSizeAGV(const TopoMeta &topoInfo)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoInfo) {
        for (const auto &podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}

void RunAllGatherVMultilevel(const TopoMeta &topoInfo, VDataDesTag vDataDes)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);

    // 算子执行参数设置
    auto rankSize = AnalyseRankSizeAGV(topoInfo);

    u64 sumRecvCount = 0;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        sumRecvCount += vDataDes.counts[rankId];
    }

    // 多线程运行ALLGATHERV算子
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(rankId);

            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendBufSize = vDataDes.counts[rankId] * sizeof(vDataDes.dataType);
            u64 recvBufSize = sumRecvCount * sizeof(vDataDes.dataType);
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclAllGatherV(sendBuf, vDataDes.counts[rankId], recvBuf, vDataDes.counts.data(),
                vDataDes.displs.data(), vDataDes.dataType, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto &thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGatherV(taskQueues, rankSize, vDataDes);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

// 2-pod (2 topo levels): 1 superpod with 2 servers, each with 2 devices
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_multilevel_2pod_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};  // 1 superpod, 2 servers, 4 devices

    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

// 2-pod (2 topo levels): 1 superpod with 2 servers, each with 3 devices
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_multilevel_2pod_6rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}}};  // 1 superpod, 2 servers, 6 devices

    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

// 3-pod topologies unsupported: simulator has only 2 net layers (no inter-superpod links at layer 2).
// Unequal AllGatherV unsupported: mesh1D slave streams end with WAIT not LOCAL_POST_TO (pre-existing bug).