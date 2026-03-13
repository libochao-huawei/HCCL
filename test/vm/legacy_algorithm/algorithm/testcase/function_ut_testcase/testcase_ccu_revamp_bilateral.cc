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
#include "task_graph_revamp_bilateral_ccu.h"
#include "check_rank_mem.h"
#include "ccu_task_transform.h"

using namespace hccl;
using namespace checker;

namespace checker{

class CcuRevampBilateralTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CcuRevampBilateralTest set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CcuRevampBilateralTest tear down" << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "CcuRevampBilateralTest set up" << std::endl;
    }

    virtual void TearDown()
    {
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

class GenBilateralStandardGraph : public GenCcuTaskNodeGraphBase {
public:
    /**-----------------------------------------Rank0------------------------------------------**/
    // ccuHead  --->  wait  --->  write  --->  post  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    /**-----------------------------------------Rank1------------------------------------------**/
    // ccuHead  --->  post  --->  wait  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    std::pair<TaskNodePtr, TaskNodePtr> GenRank1Graph_tc1(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0 = 1;
        auto waitNode = AddCcuWait(rankId, rmtRankId, queId, topicId1_0, curCcuTask);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 200);
        auto locCopy = AddCcuWrite(rankId, rmtRankId, queId, srcSlice1, dstSlice1, curCcuTask);

        uint32_t topicId0_1 = 2;
        auto postNode = AddCcuPost(rankId, rmtRankId, queId, topicId0_1, curCcuTask);

        CreateCcuEndNode(rankId, postNode, curCcuTask);

        return std::make_pair(waitNode, postNode);
    }

    std::pair<TaskNodePtr, TaskNodePtr> GenRank2Graph_tc1(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0 = 1;
        auto postNode = AddCcuPost(rankId, rmtRankId, queId, topicId1_0, curCcuTask);

        uint32_t topicId0_1 = 2;
        auto waitNode = AddCcuWait(rankId, rmtRankId, queId, topicId0_1, curCcuTask);

        CreateCcuEndNode(rankId, waitNode, curCcuTask);

        return std::make_pair(waitNode, postNode);;
    }

    TaskNodePtr GenGraph_tc1()
    {
        head = CreateHeadNode();

        auto result1 = GenRank1Graph_tc1(0, 1, 2);
        auto result2 = GenRank2Graph_tc1(1, 0, 2);

        LinkNode(result2.second, result1.first);
        LinkNode(result1.second, result2.first);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[0] = taskQeueus.get();
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[1] = taskQeueus.get();

        return head;
    }

    /**-----------------------------------------Rank0------------------------------------------**/
    // ccuHead  --->  wait  --->  write  --->  post  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    /**-----------------------------------------Rank1------------------------------------------**/
    // ccuHead  --->  post  --->  locCopy  --->  locPost  ---> locWait  --->  wait  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    std::pair<TaskNodePtr, TaskNodePtr> GenRank2Graph_tc2(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0 = 1;
        auto postNode = AddCcuPost(rankId, rmtRankId, queId, topicId1_0, curCcuTask);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 100);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 100);
        auto locCopy1 = AddCcuLocalCopy(rankId, queId, srcSlice1, dstSlice1, curCcuTask);
        
        uint32_t topicId2 = 2;
        auto locPost2 = AddCcuLocalPost(rankId, queId, topicId2, curCcuTask);
        auto locWait2 = AddCcuLocalWait(rankId, queId, topicId2, curCcuTask);

        uint32_t topicId0_1 = 3;
        auto waitNode = AddCcuWait(rankId, rmtRankId, queId, topicId0_1, curCcuTask);

        CreateCcuEndNode(rankId, waitNode, curCcuTask);

        return std::make_pair(waitNode, postNode);;
    }

    TaskNodePtr GenGraph_tc2()
    {
        head = CreateHeadNode();

        auto result1 = GenRank1Graph_tc1(0, 1, 2);
        auto result2 = GenRank2Graph_tc2(1, 0, 2);

        LinkNode(result2.second, result1.first);
        LinkNode(result1.second, result2.first);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[0] = taskQeueus.get();
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[1] = taskQeueus.get();

        return head;
    }

    /**-----------------------------------------Rank0------------------------------------------**/
    // ccuHead  --->  wait10  --->  post01  --->  write  --->  wait10  --->  post01  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    /**-----------------------------------------Rank1------------------------------------------**/
    // ccuHead  --->  post10  --->  wait01  --->  read  --->  post10  --->  wait01  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    std::vector<TaskNodePtr> GenRank1Graph_tc3(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        std::vector<TaskNodePtr> result;

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0_1 = 1;
        auto waitNode10_1 = AddCcuWait(rankId, rmtRankId, queId, topicId1_0_1, curCcuTask);
        result.push_back(waitNode10_1);

        uint32_t topicId0_1_1 = 2;
        auto postNode01_1 = AddCcuPost(rankId, rmtRankId, queId, topicId0_1_1, curCcuTask);
        result.push_back(postNode01_1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 200);
        auto ccuWrite = AddCcuWrite(rankId, rmtRankId, queId, srcSlice1, dstSlice1, curCcuTask);

        uint32_t topicId1_0_2 = 3;
        auto waitNode10_2 = AddCcuWait(rankId, rmtRankId, queId, topicId1_0_2, curCcuTask);
        result.push_back(waitNode10_2);

        uint32_t topicId0_1_2 = 4;
        auto postNode01_2 = AddCcuPost(rankId, rmtRankId, queId, topicId0_1_2, curCcuTask);
        result.push_back(postNode01_2);

        CreateCcuEndNode(rankId, postNode01_2, curCcuTask);

        return result;
    }

    std::vector<TaskNodePtr> GenRank2Graph_tc3(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        std::vector<TaskNodePtr> result;

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0_1 = 1;
        auto postNode10_1 = AddCcuPost(rankId, rmtRankId, queId, topicId1_0_1, curCcuTask);
        result.push_back(postNode10_1);

        uint32_t topicId0_1_1 = 2;
        auto waitNode01_1 = AddCcuWait(rankId, rmtRankId, queId, topicId0_1_1, curCcuTask);
        result.push_back(waitNode01_1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 200);
        auto ccuRead = AddCcuRead(rankId, rmtRankId, queId, srcSlice1, dstSlice1, curCcuTask);
        
        uint32_t topicId1_0_2 = 3;
        auto postNode10_2 = AddCcuPost(rankId, rmtRankId, queId, topicId1_0_2, curCcuTask);
        result.push_back(postNode10_2);

        uint32_t topicId0_1_2 = 4;
        auto waitNode01_2 = AddCcuWait(rankId, rmtRankId, queId, topicId0_1_2, curCcuTask);
        result.push_back(waitNode01_2);

        CreateCcuEndNode(rankId, waitNode01_2, curCcuTask);

        return result;
    }

    TaskNodePtr GenGraph_tc3()
    {
        head = CreateHeadNode();

        auto result1 = GenRank1Graph_tc3(0, 1, 2);
        auto result2 = GenRank2Graph_tc3(1, 0, 2);

        if (result1.size() != result2.size()) {
            return nullptr;
        }

        for (uint32_t i = 0; i < result1.size(); i++) {
            if (result1[i]->task->GetType() == TaskTypeStub::POST) {
                LinkNode(result1[i], result2[i]);
            } else if (result2[i]->task->GetType() == TaskTypeStub::POST) {
                LinkNode(result2[i], result1[i]);
            }
        }

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[0] = taskQeueus.get();
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[1] = taskQeueus.get();

        return head;
    }

    /**-----------------------------------------Rank0------------------------------------------**/
    // ccuHead  --->  wait  --->  write  -->  read  --->  post  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    /**-----------------------------------------Rank1------------------------------------------**/
    // ccuHead  --->  post  --->  locCopy  --->  locPost  ---> locWait  --->  wait  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    std::pair<TaskNodePtr, TaskNodePtr> GenRank1Graph_tc4(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0 = 1;
        auto waitNode = AddCcuWait(rankId, rmtRankId, queId, topicId1_0, curCcuTask);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 200, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 200, 200);
        auto ccuWrite = AddCcuWrite(rankId, rmtRankId, queId, srcSlice1, dstSlice1, curCcuTask);

        DataSlice srcSlice2(BufferType::INPUT_CCL, 400, 200);
        DataSlice dstSlice2(BufferType::OUTPUT_CCL, 300, 200);
        auto ccuRead = AddCcuWrite(rankId, rmtRankId, queId, srcSlice2, dstSlice2, curCcuTask);

        uint32_t topicId0_1 = 2;
        auto postNode = AddCcuPost(rankId, rmtRankId, queId, topicId0_1, curCcuTask);

        CreateCcuEndNode(rankId, postNode, curCcuTask);

        return std::make_pair(waitNode, postNode);
    }

    TaskNodePtr GenGraph_tc4()
    {
        head = CreateHeadNode();

        auto result1 = GenRank1Graph_tc4(0, 1, 2);
        auto result2 = GenRank2Graph_tc2(1, 0, 2);

        LinkNode(result2.second, result1.first);
        LinkNode(result1.second, result2.first);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[0] = taskQeueus.get();
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[1] = taskQeueus.get();

        return head;
    }

    /**-----------------------------------------Rank0------------------------------------------**/
    // ccuHead  --->  wait  --->  write  -->  post  -->  wait  -->  read  --->  post  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    /**-----------------------------------------Rank1------------------------------------------**/
    // ccuHead  --->  post  -->  locCopy  --->  locPost  ---> locWait  -->  wait  --->  post  --->  wait  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    std::vector<TaskNodePtr> GenRank1Graph_tc5(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        std::vector<TaskNodePtr> result;

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0_1 = 1;
        auto waitNode10_1 = AddCcuWait(rankId, rmtRankId, queId, topicId1_0_1, curCcuTask);
        result.push_back(waitNode10_1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 200, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 200, 200);
        auto ccuWrite = AddCcuWrite(rankId, rmtRankId, queId, srcSlice1, dstSlice1, curCcuTask);

        uint32_t topicId0_1_1 = 2;
        auto postNode01_1 = AddCcuPost(rankId, rmtRankId, queId, topicId0_1_1, curCcuTask);
        result.push_back(postNode01_1);

        uint32_t topicId1_0_2 = 3;
        auto waitNode10_2 = AddCcuWait(rankId, rmtRankId, queId, topicId1_0_2, curCcuTask);
        result.push_back(waitNode10_2);

        DataSlice srcSlice2(BufferType::INPUT_CCL, 400, 200);
        DataSlice dstSlice2(BufferType::OUTPUT_CCL, 300, 200);
        auto ccuRead = AddCcuWrite(rankId, rmtRankId, queId, srcSlice2, dstSlice2, curCcuTask);

        uint32_t topicId0_1_2 = 4;
        auto postNode01_2 = AddCcuPost(rankId, rmtRankId, queId, topicId0_1_2, curCcuTask);
        result.push_back(postNode01_2);

        CreateCcuEndNode(rankId, postNode01_2, curCcuTask);

        return result;
    }

    std::vector<TaskNodePtr> GenRank2Graph_tc5(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        std::vector<TaskNodePtr> result;

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId10_1 = 1;
        auto postNode10_1 = AddCcuPost(rankId, rmtRankId, queId, topicId10_1, curCcuTask);
        result.push_back(postNode10_1);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 100);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 100);
        auto locCopy1 = AddCcuLocalCopy(rankId, queId, srcSlice1, dstSlice1, curCcuTask);
        
        uint32_t topicId2 = 0xF;
        auto locPost2 = AddCcuLocalPost(rankId, queId, topicId2, curCcuTask);
        auto locWait2 = AddCcuLocalWait(rankId, queId, topicId2, curCcuTask);

        uint32_t topicId01_1 = 2;
        auto waitNode01_1 = AddCcuWait(rankId, rmtRankId, queId, topicId01_1, curCcuTask);
        result.push_back(waitNode01_1);

        uint32_t topicId01_2 = 3;
        auto postNode01_2 = AddCcuPost(rankId, rmtRankId, queId, topicId01_2, curCcuTask);
        result.push_back(postNode01_2);

        uint32_t topicId10_2 = 4;
        auto waitNode10_2 = AddCcuWait(rankId, rmtRankId, queId, topicId10_2, curCcuTask);
        result.push_back(waitNode10_2);

        CreateCcuEndNode(rankId, waitNode10_2, curCcuTask);

        return result;;
    }

    TaskNodePtr GenGraph_tc5()
    {
        head = CreateHeadNode();

        auto result1 = GenRank1Graph_tc5(0, 1, 2);
        auto result2 = GenRank2Graph_tc5(1, 0, 2);

        if (result1.size() != result2.size()) {
            return nullptr;
        }

        for (uint32_t i = 0; i < result1.size(); i++) {
            if (result1[i]->task->GetType() == TaskTypeStub::POST) {
                LinkNode(result1[i], result2[i]);
            } else if (result2[i]->task->GetType() == TaskTypeStub::POST) {
                LinkNode(result2[i], result1[i]);
            }
        }

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[0] = taskQeueus.get();
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[1] = taskQeueus.get();

        return head;
    }
};

TEST_F(CcuRevampBilateralTest, ccu_revamp_bilateral_single_success_tc1)
{
    // 生成自定义CCU子图
    GenBilateralStandardGraph genGraph;
    auto head = genGraph.GenGraph_tc1();
    std::cout<<"ccu_revamp_bilateral_single_success_tc1: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // CCU子图单边-->双边改造
    GraphRevampBilateralCcu ccuBilateralRevamp;
    ret = ccuBilateralRevamp.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_single_success_tc1: CCU Graph after parallel revamp"<<std::endl;
    genGraph.PrintRankGraph(0);
    genGraph.PrintRankGraph(1);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuRevampBilateralTest, ccu_revamp_bilateral_single_success_tc2)
{
    // 生成自定义CCU子图
    GenBilateralStandardGraph genGraph;
    auto head = genGraph.GenGraph_tc2();
    std::cout<<"ccu_revamp_bilateral_single_success_tc2: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    std::cout<<"ccu_revamp_bilateral_single_success_tc2: CCU Graph after parallel revamp"<<std::endl;

    // CCU子图单边-->双边改造
    GraphRevampBilateralCcu ccuBilateralRevamp;
    ret = ccuBilateralRevamp.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_single_success_tc2: CCU Graph after bilateral revamp"<<std::endl;
    genGraph.PrintRankGraph(0);
    genGraph.PrintRankGraph(1);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_E_MEMORY);
}

TEST_F(CcuRevampBilateralTest, ccu_revamp_bilateral_two_conflict_tc3)
{
    // 生成自定义CCU子图
    GenBilateralStandardGraph genGraph;
    auto head = genGraph.GenGraph_tc3();
    EXPECT_NE(head, nullptr);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc3: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc3: CCU Graph after parallel revamp"<<std::endl;

    // CCU子图单边-->双边改造
    GraphRevampBilateralCcu ccuBilateralRevamp;
    ret = ccuBilateralRevamp.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc3: CCU Graph after bilateral revamp"<<std::endl;
    genGraph.PrintRankGraph(0);
    genGraph.PrintRankGraph(1);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_E_MEMORY);
}

TEST_F(CcuRevampBilateralTest, ccu_revamp_bilateral_two_conflict_tc4)
{
    // 生成自定义CCU子图
    GenBilateralStandardGraph genGraph;
    auto head = genGraph.GenGraph_tc4();
    EXPECT_NE(head, nullptr);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc4: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc4: CCU Graph after parallel revamp"<<std::endl;

    // CCU子图单边-->双边改造
    GraphRevampBilateralCcu ccuBilateralRevamp;
    ret = ccuBilateralRevamp.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc4: CCU Graph after bilateral revamp"<<std::endl;
    genGraph.PrintRankGraph(0);
    genGraph.PrintRankGraph(1);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_E_MEMORY);
}

TEST_F(CcuRevampBilateralTest, ccu_revamp_bilateral_two_conflict_tc5)
{
    // 生成自定义CCU子图
    GenBilateralStandardGraph genGraph;
    auto head = genGraph.GenGraph_tc5();
    EXPECT_NE(head, nullptr);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc5: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc5: CCU Graph after parallel revamp"<<std::endl;

    // CCU子图单边-->双边改造
    GraphRevampBilateralCcu ccuBilateralRevamp;
    ret = ccuBilateralRevamp.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_two_conflict_tc5: CCU Graph after bilateral revamp"<<std::endl;
    genGraph.PrintRankGraph(0);
    genGraph.PrintRankGraph(1);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

class GenBilateralStandardGraph_1 : public GenCcuTaskNodeGraphBase {
public:
    /**-----------------------------------------Rank0------------------------------------------**/
    // ccuHead  --->  wait  --->  write  --->  post  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    /**-----------------------------------------Rank1------------------------------------------**/
    // ccuHead  --->  post  --->  wait  ---> ccuEnd
    /**----------------------------------------------------------------------------------------**/
    std::pair<TaskNodePtr, TaskNodePtr> GenRank1Graph_tc1(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0 = 1;
        auto waitNode = AddCcuWait(rankId, rmtRankId, queId, topicId1_0, curCcuTask);

        DataSlice srcSlice1(BufferType::INPUT_CCL, 0, 200);
        DataSlice dstSlice1(BufferType::OUTPUT_CCL, 0, 200);
        auto locCopy = AddCcuWrite(rankId, rmtRankId, queId, srcSlice1, dstSlice1, curCcuTask);

        uint32_t topicId0_1 = 2;
        auto postNode = AddCcuPost(rankId, rmtRankId, queId, topicId0_1, curCcuTask);

        CreateCcuEndNode(rankId, postNode, curCcuTask);

        return std::make_pair(waitNode, postNode);
    }

    std::pair<TaskNodePtr, TaskNodePtr> GenRank2Graph_tc1(RankId rankId, RankId rmtRankId, uint32_t rankSize)
    {
        uint32_t queId = 0;
        auto ccuHead = CreateCcuHeadNode(rankId, queId);
        LinkNode(head, ccuHead);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        Init(curCcuTask, rankSize, 1);
        curCcuTask->rankId = rankId;

        uint32_t topicId1_0 = 1;
        auto postNode = AddCcuPost(rankId, rmtRankId, queId, topicId1_0, curCcuTask);

        uint32_t topicId0_1 = 2;
        auto waitNode = AddCcuWait(rankId, rmtRankId, queId, topicId0_1, curCcuTask);

        CreateCcuEndNode(rankId, waitNode, curCcuTask);

        return std::make_pair(waitNode, postNode);;
    }
    TaskNodePtr GenGraph_tc1()
    {
        head = CreateHeadNode();

        auto result1 = GenRank1Graph_tc1(0, 1, 2);
        auto result2 = GenRank2Graph_tc1(1, 0, 2);

        LinkNode(result2.second, result1.first);
        LinkNode(result1.second, result2.first);

        taskQeueus = std::make_shared<SingleRankTaskQueues>();
        taskQeueus->taskQueues.resize(1);
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[0] = taskQeueus.get();
        TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[1] = taskQeueus.get();

        return head;
    }
};

TEST_F(CcuRevampBilateralTest, ccu_revamp_bilateral_single_fail_tc1)
{
    // 生成自定义CCU子图
    GenBilateralStandardGraph_1 genGraph;
    auto head = genGraph.GenGraph_tc1();

    std::cout<<"ccu_revamp_bilateral_single_test: CCU Graph before parallel revamp"<<std::endl;

    // CCU子图并行化改造
    GraphRevampParallel graphRevampParallel;
    auto ret = graphRevampParallel.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // CCU子图单边-->双边改造
    GraphRevampBilateralCcu ccuBilateralRevamp;
    ret = ccuBilateralRevamp.Revamp(head);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    std::cout<<"ccu_revamp_bilateral_single_test: CCU Graph after parallel revamp"<<std::endl;
    genGraph.PrintRankGraph(0);
    genGraph.PrintRankGraph(1);

    // 内存冲突校验
    head->hasCcuTask = true;
    CheckRankMem checkRankmem(head);
    ret = checkRankmem.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}