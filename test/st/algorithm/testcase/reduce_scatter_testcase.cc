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
#include <chrono>
#include <algorithm>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include "alg_env_config.h"
 
using namespace HcclSim;
using namespace ops_hccl;
 
class ST_REDUCE_SCATTER_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_CCU");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};
 
TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_001)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    size_t dataUnitSize = sizeof(int8_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_002)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 100 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_003)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 100 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_004)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 100 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_005)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_006)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_007)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_008)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_009)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_010)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_011)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_013)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_014)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_016)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    size_t dataUnitSize = sizeof(int16_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_017)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    size_t dataUnitSize = sizeof(int16_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_018)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    size_t dataUnitSize = sizeof(int32_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_019)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    size_t dataUnitSize = sizeof(int16_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_020)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    size_t dataUnitSize = sizeof(int8_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    // 资源清理
    SimWorld::Global()->Deinit();
}

// ============================================================
// CCU 模式 host 侧性能摸底用例
// 测量 HcclReduceScatter 在 CCU_SCHED / CCU_MS 模式下
// 每次算子下发（host 侧 planning/dispatch）的平均耗时。
// 注意：ST 框架为纯仿真（不涉及真实通信），计时反映 host 侧调度开销。
// ============================================================

namespace {
static bool IsPerfThreadPinEnabled()
{
    const char *env = getenv("HCCL_ST_PERF_PIN_THREADS");
    return env != nullptr && strcmp(env, "0") != 0;
}

static bool IsPerfRankDetailEnabled()
{
    const char *env = getenv("HCCL_ST_PERF_RANK_DETAIL");
    return env != nullptr && strcmp(env, "0") != 0;
}

// Returns 0 when the env var is not set or invalid (meaning: run all sizes).
static u64 GetPerfSizeFilterBytes()
{
    const char *env = getenv("HCCL_ST_PERF_SIZE_KIB");
    if (env == nullptr || env[0] == '\0') {
        return 0;
    }
    long kib = std::strtol(env, nullptr, 10);
    if (kib <= 0) {
        return 0;
    }
    return static_cast<u64>(kib) * 1024UL;
}

// Returns defaultCount when the env var is not set or invalid.
static int GetPerfRepeatCount(int defaultCount)
{
    const char *env = getenv("HCCL_ST_PERF_REPEATS");
    if (env == nullptr || env[0] == '\0') {
        return defaultCount;
    }
    int n = static_cast<int>(std::strtol(env, nullptr, 10));
    return (n > 0) ? n : defaultCount;
}

static std::vector<int> GetAllowedCpuList()
{
    std::vector<int> cpuList;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        return cpuList;
    }

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            cpuList.push_back(cpu);
        }
    }
    return cpuList;
}

static void TryPinCurrentThreadToRankCpu(int rankId)
{
    if (!IsPerfThreadPinEnabled()) {
        return;
    }

    const auto cpuList = GetAllowedCpuList();
    if (cpuList.empty()) {
        return;
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpuList[rankId % cpuList.size()], &mask);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static const std::vector<u64>& GetSmallPerfMessageBytes()
{
    static const std::vector<u64> messageBytes {
        2UL * 1024,
        4UL * 1024,
        8UL * 1024,
        16UL * 1024,
        32UL * 1024,
        64UL * 1024,
        128UL * 1024,
        256UL * 1024,
        512UL * 1024,
    };
    return messageBytes;
}

// Returns a (possibly single-element) list respecting HCCL_ST_PERF_SIZE_KIB.
static std::vector<u64> GetActivePerfMessageBytes()
{
    const u64 filterBytes = GetPerfSizeFilterBytes();
    if (filterBytes == 0) {
        return GetSmallPerfMessageBytes();
    }
    // Validate that requested size is in the standard sweep table.
    for (u64 b : GetSmallPerfMessageBytes()) {
        if (b == filterBytes) {
            return {filterBytes};
        }
    }
    // Not in table — accept it anyway so the user can probe any power-of-two.
    return {filterBytes};
}

struct PerfDistribution {
    double minUs {0.0};
    double medianUs {0.0};
    double p95Us {0.0};
    double maxUs {0.0};
};

struct PerfRepeatResult {
    std::vector<double> rankTotalUs;
    std::vector<std::vector<double>> rankIterUs;
    double bottleneckUs {0.0};
    int bottleneckRank {-1};
};

struct PerfIterSpanInfo {
    double spanUs {0.0};
    int repeatId {-1};
    int iterId {-1};
    int slowestRank {-1};
    int fastestRank {-1};
    double slowestUs {0.0};
    double fastestUs {0.0};
};

static double CalcMedianFromSorted(const std::vector<double>& values)
{
    const size_t size = values.size();
    if (size == 0) {
        return 0.0;
    }
    const size_t mid = size / 2;
    if ((size % 2) == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
}

static double CalcPercentileFromSorted(const std::vector<double>& values, double ratio)
{
    if (values.empty()) {
        return 0.0;
    }
    const double scaledIndex = ratio * static_cast<double>(values.size() - 1);
    const size_t index = static_cast<size_t>(scaledIndex);
    return values[index];
}

static PerfDistribution CalcPerfDistribution(std::vector<double> values)
{
    PerfDistribution dist;
    if (values.empty()) {
        return dist;
    }

    std::sort(values.begin(), values.end());
    dist.minUs = values.front();
    dist.medianUs = CalcMedianFromSorted(values);
    dist.p95Us = CalcPercentileFromSorted(values, 0.95);
    dist.maxUs = values.back();
    return dist;
}

static PerfRepeatResult RunCcuReduceScatterPerfOnce(
    const char* modeName,
    const TopoMeta& topoMeta,
    u64 recvCount,
    HcclDataType dataType,
    size_t dataUnitSize,
    HcclReduceOp reduceOp,
    int warmupIters,
    int perfIters)
{
    PerfRepeatResult result;
    auto rankSize = 0;
    for (const auto& pod : topoMeta[0]) { rankSize += pod.size(); }

    result.rankTotalUs.assign(rankSize, 0.0);
    result.rankIterUs.assign(rankSize, std::vector<double>(perfIters, 0.0));

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", modeName, 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_ENABLE_OPEN_CCU", "1", 1);

    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=, &result]() {
            TryPinCurrentThreadToRankCpu(rankId);
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            aclrtMalloc(&sendBuf, recvCount * dataUnitSize * rankSize,
                static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvCount * dataUnitSize,
                static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            for (int j = 0; j < warmupIters; ++j) {
                CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < perfIters; ++i) {
                auto iterT0 = std::chrono::high_resolution_clock::now();
                CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
                auto iterT1 = std::chrono::high_resolution_clock::now();
                result.rankIterUs[rankId][i] =
                    std::chrono::duration<double, std::micro>(iterT1 - iterT0).count();
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            result.rankTotalUs[rankId] = std::chrono::duration<double, std::micro>(t1 - t0).count();

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto& t : threads) { t.join(); }

    auto bottleneckIt = std::max_element(result.rankTotalUs.begin(), result.rankTotalUs.end());
    result.bottleneckUs = *bottleneckIt;
    result.bottleneckRank = static_cast<int>(std::distance(result.rankTotalUs.begin(), bottleneckIt));

    SimWorld::Global()->Deinit();
    return result;
}

// CCU 模式性能测试辅助函数
// modeName    : "CCU_SCHED" 或 "CCU_MS"
// topoMeta    : 超节点-Server-Device 三维拓扑
// recvCount   : 每 rank 接收元素数
// dataUnitSize: 每元素字节数
// warmupIters : 预热迭代次数（不计时）
// perfIters   : 计时迭代次数
static void RunCcuReduceScatterPerf(
    const char* modeName,
    const TopoMeta& topoMeta,
    u64 recvCount,
    HcclDataType dataType,
    size_t dataUnitSize,
    HcclReduceOp reduceOp,
    int warmupIters,
    int perfIters,
    int perfRepeats)
{
    auto rankSize = 0;
    for (const auto& pod : topoMeta[0]) { rankSize += pod.size(); }
    const u64 messageBytes = recvCount * dataUnitSize;

    std::vector<double> repeatBottleneckAvgUs;
    repeatBottleneckAvgUs.reserve(perfRepeats);
    std::vector<int> repeatBottleneckCounts(rankSize, 0);
    std::vector<int> iterSlowestCounts(rankSize, 0);
    std::vector<std::vector<double>> rankIterSamples(rankSize);
    std::vector<double> iterSpanSamples;
    iterSpanSamples.reserve(static_cast<size_t>(perfRepeats) * perfIters);
    PerfIterSpanInfo worstIterSpan;
    worstIterSpan.spanUs = -1.0;

    for (int repeatId = 0; repeatId < perfRepeats; ++repeatId) {
        PerfRepeatResult repeat = RunCcuReduceScatterPerfOnce(
            modeName, topoMeta, recvCount, dataType, dataUnitSize, reduceOp, warmupIters, perfIters);

        const double repeatAvgUs = repeat.bottleneckUs / perfIters;
        repeatBottleneckAvgUs.push_back(repeatAvgUs);
        ++repeatBottleneckCounts[repeat.bottleneckRank];

        if (IsPerfRankDetailEnabled()) {
            printf("[PERF_REPEAT][%s] repeat=%d/%d bytes=%lu (%lu KiB) recvCount=%lu bottleneckRank=%d avg=%.2f us/iter\n",
                modeName,
                repeatId + 1,
                perfRepeats,
                static_cast<unsigned long>(messageBytes),
                static_cast<unsigned long>(messageBytes / 1024),
                static_cast<unsigned long>(recvCount),
                repeat.bottleneckRank,
                repeatAvgUs);
        }

        for (int rankId = 0; rankId < rankSize; ++rankId) {
            const auto& iterUs = repeat.rankIterUs[rankId];
            rankIterSamples[rankId].insert(rankIterSamples[rankId].end(), iterUs.begin(), iterUs.end());
        }

        for (int iterId = 0; iterId < perfIters; ++iterId) {
            double fastestUs = std::numeric_limits<double>::max();
            double slowestUs = std::numeric_limits<double>::lowest();
            int fastestRank = -1;
            int slowestRank = -1;
            for (int rankId = 0; rankId < rankSize; ++rankId) {
                const double iterUs = repeat.rankIterUs[rankId][iterId];
                if (iterUs < fastestUs) {
                    fastestUs = iterUs;
                    fastestRank = rankId;
                }
                if (iterUs > slowestUs) {
                    slowestUs = iterUs;
                    slowestRank = rankId;
                }
            }

            ++iterSlowestCounts[slowestRank];
            const double spanUs = slowestUs - fastestUs;
            iterSpanSamples.push_back(spanUs);
            if (spanUs > worstIterSpan.spanUs) {
                worstIterSpan.spanUs = spanUs;
                worstIterSpan.repeatId = repeatId;
                worstIterSpan.iterId = iterId;
                worstIterSpan.slowestRank = slowestRank;
                worstIterSpan.fastestRank = fastestRank;
                worstIterSpan.slowestUs = slowestUs;
                worstIterSpan.fastestUs = fastestUs;
            }
        }
    }

    const PerfDistribution repeatDist = CalcPerfDistribution(repeatBottleneckAvgUs);
    printf("[PERF][%s] rankSize=%d bytes=%lu (%lu KiB) recvCount=%lu FP32 warmup=%d iters=%d repeats=%d  min=%.2f median=%.2f max=%.2f us/iter\n",
        modeName,
        rankSize,
        static_cast<unsigned long>(messageBytes),
        static_cast<unsigned long>(messageBytes / 1024),
        static_cast<unsigned long>(recvCount),
        warmupIters,
        perfIters,
        perfRepeats,
        repeatDist.minUs,
        repeatDist.medianUs,
        repeatDist.maxUs);

    if (IsPerfRankDetailEnabled()) {
        for (int rankId = 0; rankId < rankSize; ++rankId) {
            const PerfDistribution rankDist = CalcPerfDistribution(rankIterSamples[rankId]);
            printf("[PERF_RANK_DIST][%s] bytes=%lu (%lu KiB) rank=%d repeatBottleneck=%d/%d iterSlowest=%d/%d  iter_min=%.2f iter_p50=%.2f iter_p95=%.2f iter_max=%.2f us\n",
                modeName,
                static_cast<unsigned long>(messageBytes),
                static_cast<unsigned long>(messageBytes / 1024),
                rankId,
                repeatBottleneckCounts[rankId],
                perfRepeats,
                iterSlowestCounts[rankId],
                perfRepeats * perfIters,
                rankDist.minUs,
                rankDist.medianUs,
                rankDist.p95Us,
                rankDist.maxUs);
        }

        const PerfDistribution spanDist = CalcPerfDistribution(iterSpanSamples);
        printf("[PERF_ITER_SPAN][%s] bytes=%lu (%lu KiB) span_min=%.2f span_p50=%.2f span_p95=%.2f span_max=%.2f us\n",
            modeName,
            static_cast<unsigned long>(messageBytes),
            static_cast<unsigned long>(messageBytes / 1024),
            spanDist.minUs,
            spanDist.medianUs,
            spanDist.p95Us,
            spanDist.maxUs);
        printf("[PERF_ITER_WORST][%s] bytes=%lu (%lu KiB) repeat=%d iter=%d slowestRank=%d slowest=%.2f us fastestRank=%d fastest=%.2f us span=%.2f us\n",
            modeName,
            static_cast<unsigned long>(messageBytes),
            static_cast<unsigned long>(messageBytes / 1024),
            worstIterSpan.repeatId + 1,
            worstIterSpan.iterId,
            worstIterSpan.slowestRank,
            worstIterSpan.slowestUs,
            worstIterSpan.fastestRank,
            worstIterSpan.fastestUs,
            worstIterSpan.spanUs);
    }
    fflush(stdout);
}
} // anonymous namespace

// CCU_SCHED 模式：调度器驱动 CCU，小数据量性能摸底（单 server 内 8 卡）
TEST_F(ST_REDUCE_SCATTER_TEST, test_ccu_sched_reducescatter_perf_001)
{
    // 拓扑：1 超节点 × 1 server × 8 device = 8 ranks
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};
    const int warmupIters = 10;
    const int perfIters   = 200;
    const int perfRepeats = GetPerfRepeatCount(5);
    // 小数据量梯度测试：2 KiB 到 512 KiB（可通过 HCCL_ST_PERF_SIZE_KIB 限定单个 size）
    for (u64 messageBytes : GetActivePerfMessageBytes()) {
        u64 recvCount = messageBytes / sizeof(float);
        RunCcuReduceScatterPerf(
            "CCU_SCHED", topoMeta, recvCount,
            HcclDataType::HCCL_DATA_TYPE_FP32, sizeof(float),
            HcclReduceOp::HCCL_REDUCE_SUM,
            warmupIters, perfIters, perfRepeats);
    }
}

// CCU_MS 模式：多流驱动 CCU，小数据量性能摸底（单 server 内 8 卡）
TEST_F(ST_REDUCE_SCATTER_TEST, test_ccu_ms_reducescatter_perf_001)
{
    // 拓扑：1 超节点 × 1 server × 8 device = 8 ranks
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};
    const int warmupIters = 10;
    const int perfIters   = 200;
    const int perfRepeats = GetPerfRepeatCount(5);
    // 小数据量梯度测试：2 KiB 到 512 KiB（可通过 HCCL_ST_PERF_SIZE_KIB 限定单个 size）
    for (u64 messageBytes : GetActivePerfMessageBytes()) {
        u64 recvCount = messageBytes / sizeof(float);
        RunCcuReduceScatterPerf(
            "CCU_MS", topoMeta, recvCount,
            HcclDataType::HCCL_DATA_TYPE_FP32, sizeof(float),
            HcclReduceOp::HCCL_REDUCE_SUM,
            warmupIters, perfIters, perfRepeats);
    }
}

// 单线程微基准：只测 host 计时本身的上限能力，不引入 8 线程并发抢占。
TEST_F(ST_REDUCE_SCATTER_TEST, test_single_thread_single_rank_micro_bench_001)
{
    constexpr int benchIters = 200000;
    std::vector<double> emptySpanUs;
    std::vector<double> callSpanUs;
    emptySpanUs.reserve(benchIters);
    callSpanUs.reserve(benchIters);

    TryPinCurrentThreadToRankCpu(0);

    auto noOpCall = []() -> HcclResult {
        return HCCL_SUCCESS;
    };

    for (int i = 0; i < benchIters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();
        emptySpanUs.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

        auto c0 = std::chrono::high_resolution_clock::now();
        ASSERT_EQ(noOpCall(), HCCL_SUCCESS);
        auto c1 = std::chrono::high_resolution_clock::now();
        callSpanUs.push_back(std::chrono::duration<double, std::micro>(c1 - c0).count());
    }

    const PerfDistribution emptyDist = CalcPerfDistribution(emptySpanUs);
    const PerfDistribution callDist = CalcPerfDistribution(callSpanUs);
    printf("[PERF_MICRO][SINGLE_THREAD_SINGLE_RANK] iters=%d "
        "empty_p50=%.6f empty_p95=%.6f empty_max=%.6f us "
        "noop_p50=%.6f noop_p95=%.6f noop_max=%.6f us\n",
        benchIters,
        emptyDist.medianUs,
        emptyDist.p95Us,
        emptyDist.maxUs,
        callDist.medianUs,
        callDist.p95Us,
        callDist.maxUs);
    fflush(stdout);
}

// 单线程微基准：真实 HcclReduceScatter 调用（8卡环境仅记录rank 0的调用耗时）
// 分三阶段：
//   Phase 1: 多线程初始化（所有8个rank创建通信域）
//   Phase 2: 单线程单卡计时（仅rank 0在此时执行）
//   Phase 3: 多线程清理/join（所有rank销毁通信域）
// 用于与 test_single_thread_single_rank_micro_bench_001 对比，
// 看实际算子开销相对于纯计时开销的差异。
TEST_F(ST_REDUCE_SCATTER_TEST, test_single_thread_single_rank_micro_bench_with_reducescatter_002)
{
    constexpr int benchIters = 200;
    constexpr int rankSize = 8;  // 8 卡环境
    constexpr int testRankId = 0;  // 仅在rank 0计时
    const u64 recvCount = 256UL * 1024 / sizeof(float);  // 256 KiB
    const HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    const size_t dataUnitSize = sizeof(float);
    const HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;

    std::vector<double> rank0RsSpanUs;
    rank0RsSpanUs.reserve(benchIters);

    // 初始化仿真环境（8卡）
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_ENABLE_OPEN_CCU", "1", 1);

    // Singleton barrier for phase synchronization
    std::atomic<int> barrierCount(0);
    std::condition_variable barrierCv;
    std::mutex barrierMutex;

    auto barrierWait = [&](int tid, int expectedCount) {
        std::unique_lock<std::mutex> lock(barrierMutex);
        ++barrierCount;
        if (barrierCount < expectedCount) {
            barrierCv.wait(lock, [&] { return barrierCount >= expectedCount; });
        } else {
            barrierCv.notify_all();
        }
    };

    // 多线程执行 Phase 1(初始化) + Phase 2(计时rankId0) + Phase 3(清理)
    std::vector<std::thread> threads;
    for (int rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([rankId, rankSize, recvCount, dataUnitSize, dataType, reduceOp,
                              testRankId, benchIters, &rank0RsSpanUs, &barrierWait]() {
            TryPinCurrentThreadToRankCpu(rankId);
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            auto ret = HcclCommInitClusterInfo("./ranktable.json", rankId, &comm);
            if (ret != HCCL_SUCCESS) {
                return;
            }

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            aclrtMalloc(&sendBuf, recvCount * dataUnitSize * rankSize,
                static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvCount * dataUnitSize,
                static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 预热 10 次
            for (int j = 0; j < 10; ++j) {
                HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream);
            }

            // Phase barrier: 所有rank预热完成后开始计时相位
            barrierWait(rankId, rankSize);

            // Phase 2: 仅rank 0计时，其他rank无计时直接调用
            if (rankId == testRankId) {
                for (int i = 0; i < benchIters; ++i) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    rank0RsSpanUs.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
            } else {
                // 其他rank无计时直接调用 200k 次
                for (int i = 0; i < benchIters; ++i) {
                    HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream);
                }
            }

            // Phase barrier: 所有rank计时均完成后开始清理相位
            barrierWait(rankId, rankSize);

            // Phase 3: 清理
            HcclCommDestroy(comm);
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) { t.join(); }

    const PerfDistribution rsDist = CalcPerfDistribution(rank0RsSpanUs);
    printf("[PERF_MICRO][SINGLE_THREAD_SINGLE_RANK_RS] iters=%d rankSize=%d testRank=%d recvCount=%lu bytes=%lu "
        "rs_p50=%.6f rs_p95=%.6f rs_max=%.6f us\n",
        benchIters,
        rankSize,
        testRankId,
        static_cast<unsigned long>(recvCount),
        static_cast<unsigned long>(recvCount * dataUnitSize),
        rsDist.medianUs,
        rsDist.p95Us,
        rsDist.maxUs);
    fflush(stdout);

    SimWorld::Global()->Deinit();
}

