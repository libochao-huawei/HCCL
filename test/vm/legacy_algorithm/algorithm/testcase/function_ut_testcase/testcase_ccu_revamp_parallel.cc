/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include <vector>
#include <iostream>
#include <string>

#include "coll_service_stub.h"
#include "checker.h"
#include "testcase_utils.h"
#include "topo_meta.h"
#include "gen_ccu_task_node_utils.h"
#include "task_graph_revamp_parallel.h"
#include "check_rank_mem.h"
#include "ccu_task_transform.h"
#include "ccu_task_common.h"

using namespace hccl;
using namespace checker;

namespace checker{

class CcuRevampParallelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CcuRevampParallelTest set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CcuRevampParallelTest tear down" << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "CcuRevampParallelTest set up" << std::endl;
    }

    virtual void TearDown()
    {
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

class GenAsyncStandardGraph : public GenCcuTaskNodeGraphBase {
public:
    
    // ccuHead  --->  locCopy1  --->  locPost1 --->  locWait1  ---> ccuEnd
    TaskNodePtr GenGraph1()
    {
        head = CreateHeadNode();
        RankId rankId = 0;
        uint32_t queueId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queueId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, 1, 1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 100);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 100);
        AddCcuLocalCopy(rankId, queueId, srcSlice1, dstSlice1, curCcuTask);
        
        uint32_t topicId2 = 1;
        auto locPost2 = AddCcuLocalPost(rankId, queueId, topicId2, curCcuTask);
        auto locWait2 = AddCcuLocalWait(rankId, queueId, topicId2, curCcuTask);

        CreateCcuEndNode(rankId, locWait2, curCcuTask);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[rankId] = taskQeueus.get();

        return head;
    }

    // ccuHead  --->  locCopy1  --->  locPost1 --->  locCopy2 --->  locPost2 --->  locWait1 --->  locWait2  ---> ccuEnd
    //                                locPost1 --------------------------------->  locWait1
    //                                                              locPost2  ----------------->  locWait2
    TaskNodePtr GenGraph2()
    {
        head = CreateHeadNode();
        RankId rankId = 0;
        uint32_t queueId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queueId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, 1, 1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 100);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 100);
        AddCcuLocalCopy(rankId, queueId, srcSlice1, dstSlice1, curCcuTask);
        
        uint32_t topicId2 = 1;
        auto locPost2 = AddCcuLocalPost(rankId, queueId, topicId2, curCcuTask);

        DataSlice srcSlice2(BufferType::INPUT_CCL, 200, 200);
        DataSlice dstSlice2(BufferType::OUTPUT_CCL, 0, 200);
        AddCcuLocalCopy(rankId, queueId, srcSlice2, dstSlice2, curCcuTask);

        uint32_t topicId3 = 2;
        auto locPost3 = AddCcuLocalPost(rankId, queueId, topicId3, curCcuTask);

        auto locWait2 = AddCcuLocalWait(rankId, queueId, topicId2, curCcuTask);
        LinkNode(locPost2, locWait2);

        auto locWait3 = AddCcuLocalWait(rankId, queueId, topicId3, curCcuTask);
        LinkNode(locPost3, locWait3);

        CreateCcuEndNode(rankId, locWait3, curCcuTask);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[rankId] = taskQeueus.get();

        return head;
    }

    // ccuHead  --->  locCopy1  --->  locPost1 --->  locWait1 --->  locCopy2 --->  locPost2 --->  locWait2  ---> ccuEnd
    TaskNodePtr GenGraph3()
    {
        head = CreateHeadNode();
        RankId rankId = 0;
        uint32_t queueId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queueId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, 1, 1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 100);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 100);
        AddCcuLocalCopy(rankId, queueId, srcSlice1, dstSlice1, curCcuTask);
        
        uint32_t topicId2 = 1;
        auto locPost2 = AddCcuLocalPost(rankId, queueId, topicId2, curCcuTask);
        auto locWait2 = AddCcuLocalWait(rankId, queueId, topicId2, curCcuTask);

        DataSlice srcSlice2(BufferType::INPUT_CCL, 200, 200);
        DataSlice dstSlice2(BufferType::OUTPUT_CCL, 0, 200);
        AddCcuLocalCopy(rankId, queueId, srcSlice2, dstSlice2, curCcuTask);

        uint32_t topicId3 = 2;
        auto locPost3 = AddCcuLocalPost(rankId, queueId, topicId3, curCcuTask);
        auto locWait3 = AddCcuLocalWait(rankId, queueId, topicId3, curCcuTask);

        CreateCcuEndNode(rankId, locWait3, curCcuTask);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[rankId] = taskQeueus.get();

        return head;
    }

    // ccuHead  --->  locCopy1  --->  locCopy2 --->  locPost --->  locWait  ---> ccuEnd
    TaskNodePtr GenGraph4()
    {
        head = CreateHeadNode();
        RankId rankId = 0;
        uint32_t queueId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queueId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, 1, 1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 100);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 100);
        AddCcuLocalCopy(rankId, queueId, srcSlice1, dstSlice1, curCcuTask);

        DataSlice srcSlice2(BufferType::INPUT_CCL, 200, 200);
        DataSlice dstSlice2(BufferType::OUTPUT_CCL, 0, 200);
        AddCcuLocalCopy(rankId, queueId, srcSlice2, dstSlice2, curCcuTask);

        uint32_t topicId = 1;
        auto locPost = AddCcuLocalPost(rankId, queueId, topicId, curCcuTask);
        auto locWait = AddCcuLocalWait(rankId, queueId, topicId, curCcuTask);

        CreateCcuEndNode(rankId, locWait, curCcuTask);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[rankId] = taskQeueus.get();

        return head;
    }
};

TEST_F(CcuRevampParallelTest, ccu_mem_conflict_for_single_async_node)
{
    // 生成自定义CCU子图
    GenAsyncStandardGraph genGraph;
    auto head = genGraph.GenGraph1();

    std::cout<<"ccu_mem_conflict_for_single_async_node: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_mem_conflict_for_single_async_node: CCU Graph after parallel revamp"<<std::endl;
    genGraph.PrintRankGraph(0);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuRevampParallelTest, ccu_mem_conflict_for_two_continuous_async_node)
{
    // 生成自定义CCU子图
    GenAsyncStandardGraph genGraph;
    auto head = genGraph.GenGraph2();
    genGraph.PrintRankGraph(0);

    std::cout<<"ccu_mem_conflict_for_two_continuous_async_node: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    // EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_mem_conflict_for_two_continuous_async_node: CCU Graph after parallel revamp"<<std::endl;
    genGraph.PrintRankGraph(0);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    // EXPECT_EQ(ret, HcclResult::HCCL_E_MEMORY);
}

TEST_F(CcuRevampParallelTest, ccu_mem_conflict_for_two_discontinuous_async_node_success)
{
    // 生成自定义CCU子图
    GenAsyncStandardGraph genGraph;
    auto head = genGraph.GenGraph3();

    std::cout<<"ccu_mem_conflict_for_two_discontinuous_async_node_success: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_mem_conflict_for_two_discontinuous_async_node_success: CCU Graph after parallel revamp"<<std::endl;
    genGraph.PrintRankGraph(0);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuRevampParallelTest, ccu_mem_conflict_for_two_discontinuous_async_node_conflict)
{
    // 生成自定义CCU子图
    GenAsyncStandardGraph genGraph;
    auto head = genGraph.GenGraph4();

    std::cout<<"ccu_mem_conflict_for_two_discontinuous_async_node_conflict: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_mem_conflict_for_two_discontinuous_async_node_conflict: CCU Graph after parallel revamp"<<std::endl;

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_E_MEMORY);
}

class GenLoopStandardGraph : public GenCcuTaskNodeGraphBase {
public:
    std::pair<TaskNodePtr, TaskNodePtr> GenLoopBlock(uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, TaskStubCcuGraph *curCcuTask)
    {
        auto rankId = curCcuTask->GetRankId();
        auto loopStart = Hccl::AddLoopStartTask(queId, loopIdx, loopGroupIdx, curCcuTask);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 100);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 100);
        AddCcuLocalCopy(rankId, queId, srcSlice1, dstSlice1, curCcuTask);

        uint32_t topicId = 1 + loopIdx;
        auto locPost = AddCcuLocalPost(rankId, queId, topicId, curCcuTask);
        auto locWait = AddCcuLocalWait(rankId, queId, topicId, curCcuTask);

        auto loopEnd = Hccl::AddLoopEndTask(queId, loopIdx, loopGroupIdx, curCcuTask);
        return std::make_pair(loopStart, loopEnd);
    }
};

class GenOneLoopStandardGraph : public GenLoopStandardGraph {
public:
    /**-----------------------------------------Rank0------------------------------------------**/
    // LoopBlock:  loopStart  --->  locCopy1  --->  locPost1  --->  locWait1  --->  loopEnd
    // ccuHead  --->  LoopBlock1  --->  locCopy2  --->  locPost2 --->  locWait2  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    TaskNodePtr GenGraph()
    {
        head = CreateHeadNode();
        RankId rankId = 0;
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, 1, 1);
        curCcuTask->rankId = rankId;

        // ccu子图设置：1个loopGroup，且其对应展开后有2个loop指令块
        curCcuTask->loopGroupIdx = 1; // 设置loopGroup个数
        curCcuTask->loopIdx[0] = 1;// 设置每个loopGroup展开后的loop个数

        auto loopBlock1 = GenLoopBlock(queId, 0, 0, curCcuTask);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 100, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 100, 200);
        AddCcuLocalCopy(rankId, queId, srcSlice1, dstSlice1, curCcuTask);
        auto locPostLast = AddCcuLocalPost(rankId, queId, 0xFF, curCcuTask);
        auto locWaitLast = AddCcuLocalWait(rankId, queId, 0xFF, curCcuTask);

        CreateCcuEndNode(rankId, locWaitLast, curCcuTask);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[rankId] = taskQeueus.get();

        return head;
    }
};

TEST_F(CcuRevampParallelTest, ccu_mem_conflict_for_1_loop_node)
{
    // 生成自定义CCU子图
    GenOneLoopStandardGraph genGraph;
    auto head = genGraph.GenGraph();

    std::cout<<"ccu_mem_conflict_for_loop_node: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_mem_conflict_for_loop_node: CCU Graph after parallel revamp"<<std::endl;

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

class GenTwoLoopStandardGraph : public GenLoopStandardGraph {
public:
    /**-----------------------------------------Rank0------------------------------------------**/
    // LoopBlock:  loopStart  --->  locCopy1  --->  locPost1  --->  locWait1  --->  loopEnd
    // ccuHead  --->  LoopBlock1  --->  LoopBlock2  --->  locCopy2  --->  locPost2 --->  locWait2  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    TaskNodePtr GenGraph()
    {
        head = CreateHeadNode();
        RankId rankId = 0;
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, 1, 1);
        curCcuTask->rankId = rankId;

        // ccu子图设置：1个loopGroup，且其对应展开后有2个loop指令块
        curCcuTask->loopGroupIdx = 1; // 设置loopGroup个数
        curCcuTask->loopIdx[0] = 2;// 设置每个loopGroup展开后的loop个数

        auto loopBlock1 = GenLoopBlock(queId, 0, 0, curCcuTask);
        auto loopBlock2 = GenLoopBlock(queId, 1, 0, curCcuTask);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 100, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 100, 200);
        AddCcuLocalCopy(rankId, queId, srcSlice1, dstSlice1, curCcuTask);
        auto locPostLast = AddCcuLocalPost(rankId, queId, 0xFF, curCcuTask);
        auto locWaitLast = AddCcuLocalWait(rankId, queId, 0xFF, curCcuTask);

        CreateCcuEndNode(rankId, locWaitLast, curCcuTask);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[rankId] = taskQeueus.get();

        return head;
    }
};

TEST_F(CcuRevampParallelTest, ccu_mem_conflict_for_2_loop_node)
{
    // 生成自定义CCU子图
    GenTwoLoopStandardGraph genGraph;
    auto head = genGraph.GenGraph();

    std::cout<<"ccu_mem_conflict_for_loop_node: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_mem_conflict_for_loop_node: CCU Graph after parallel revamp"<<std::endl;
    genGraph.PrintRankGraph(0);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_E_MEMORY);
}

}