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
#include "alg/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"
 
using namespace HcclSim;
using namespace ops_hccl;
 
class ST_ALL2ALLV_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};
 
TEST_F(ST_ALL2ALLV_TEST, st_all2allV)
{
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
 
    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
 
    uint32_t rankSize = 4;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    std::vector<u64> sendCountMatrix = {
        100, 100, 100, 100,
        100, 100, 100, 100,
        100, 100, 100, 100,
        100, 100, 100, 100,
    };
    std::vector<u64> recvCounts = {
        100, 100, 100, 100,
        100, 100, 100, 100,
        100, 100, 100, 100,
        100, 100, 100, 100,
    };
    std::vector<u64> sdispls = {
        0, 100, 200, 300,
        0, 100, 200, 300,
        0, 100, 200, 300,
        0, 100, 200, 300,
    };
    std::vector<u64> rdispls = {
        0, 100, 200, 300,
        0, 100, 200, 300,
        0, 100, 200, 300,
        0, 100, 200, 300,
    };
 
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
 
            u64 sendDataCount = 0;  // 数据量转化为字节数
            for (int i = 0; i < rankSize; ++i) {
                sendDataCount += sendCountMatrix[rankId * rankSize + i];
            }
            u64 recvDataCount = 0;
            for (int i = 0; i < rankSize; ++i) {
                recvDataCount += recvCounts[rankId * rankSize + i];
            }
 
            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendDataCount * sizeof(float), static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvDataCount * sizeof(float), static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
 
            // 4.算子下发
            CHK_RET(HcclAlltoAllV(sendBuf,
                &sendCountMatrix[rankId * rankSize],  // 当前 rank 的发送计数
                &sdispls[rankId * rankSize],     // 当前 rank 的发送位移
                dataType,
                recvBuf,
                &recvCounts[rankId * rankSize],  // 当前 rank 的接收计数
                &rdispls[rankId * rankSize],     // 当前 rank 的接收位移
                dataType,
                comm,
                stream));
 
            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
 
    // 等待多线程执行完成
    for (auto& thread : threads) {
        thread.join();
    }
 
    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAll2AllV(taskQueues, rankSize, dataType, sendCountMatrix);
    EXPECT_TRUE(res == HCCL_SUCCESS);
}