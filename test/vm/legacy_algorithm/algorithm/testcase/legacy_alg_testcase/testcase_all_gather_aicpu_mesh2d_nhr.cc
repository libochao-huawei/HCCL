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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include <vector>
#include <iostream>
#include <string>

#define private public
#define protected public
#include "coll_service_stub.h"
#include "checker.h"
#include "testcase_utils.h"
#include "topo_meta.h"

#include "dev_buffer.h"
#include "virtual_topo_stub.h"
#include "virtual_topo.h"
#include "topo_match_concurr_mesh_nhr.h"
#include "ins_temp_all_gather_mesh_2D.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_all_gather_parallel_executor.h"
#include "template_utils.h"
#include "ins_exe_que.h"
#include "data_type.h"
#include "dev_buffer.h"
#include <memory>
#include "primitive.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "log.h"
#include "ins_queue.h"
#include <chrono>

using namespace Hccl;
namespace checker {
class AllGatherAICPUMesh2dNHRTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGather AICPU ParrallelMesh2DNHR test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGather AICPU ParrallelMesh2DNHR test tear down" << std::endl;
    }

    virtual void SetUp()
    {
        const ::testing::TestInfo *const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName =
            "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
        Checker::SetDumpFileName(caseName);
    }

    virtual void TearDown()
    {
        Checker::SetDumpFileName("analysis_result");
        GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};
}  // namespace checker

TEST_F(AllGatherAICPUMesh2dNHRTest, Aicpu_AllGather_Mesh_template)
{
    // 创建需求资源
    RankId myRank = 0;
    u32 rankSize = 8;
    std::vector<std::vector<RankId>> tempVTopo = {{0, 1}, {0, 2}};
    std::map<RankId, u32> tempVirtRankMap = {{{0, 0}, {1, 1}, {2, 2}, {3, 3}}};
    u32 rankLevel0Size = tempVirtRankMap.size();
    // 开始执行
    std::shared_ptr<InsTempAllGatherMesh2D> algoTemplate =
        std::make_shared<InsTempAllGatherMesh2D>(myRank, rankLevel0Size, tempVTopo, tempVirtRankMap);

    // 结果验证
    EXPECT_NE(algoTemplate, nullptr);
    AlgTempResReq tempResReq;

    // 开始执行
    ASSERT_EQ(algoTemplate->CalcRes(tempResReq), HcclResult::HCCL_SUCCESS);
}

// {{0,1},{8,9},{24,25},{32,33}},{{3,5},{19,21}},非对称2D
TEST_F(AllGatherAICPUMesh2dNHRTest, Aicpu_AllGather_Parallel_executor_4x2_2x2)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910_95));

    RankId myRank = 8;
    u32 rankSize = 12;

    VirtualTopoStub virtTopo(myRank);
    string rankTable = "test";
    virtTopo.TopoInit91095TwoPodFourTwoAndTwoTwo(rankTable);

    std::unique_ptr<InsAllGatherParallelExecutor<TopoMatchConcurrMeshNHR, InsTempAllGatherMesh2D, InsTempAllGatherNHR>>
        algoExecutor(
            new InsAllGatherParallelExecutor<TopoMatchConcurrMeshNHR, InsTempAllGatherMesh2D, InsTempAllGatherNHR>);
    algoExecutor->SetMyRank(myRank);
    algoExecutor->SetRankSize(rankSize);
    algoExecutor->EnableDataAllign(false);
    algoExecutor->EnableDetour(false);
    algoExecutor->SetDevType(DevType::DEV_TYPE_910_95);

    CollAlgOperator collAlgOp;
    collAlgOp.opType = OpType::ALLGATHER;
    collAlgOp.dataCount = 600;
    collAlgOp.dataType = DataType::FP32;
    uint64_t dataSize = collAlgOp.dataCount * 2;

    collAlgOp.inputMem = DevBuffer::Create(0x1000000, dataSize);
    collAlgOp.outputMem = DevBuffer::Create(0x2000000, dataSize * rankSize);
    collAlgOp.scratchMem = DevBuffer::Create(0x3000000, dataSize * rankSize);

    CollAlgParams collAlgParams;
    collAlgParams.opMode = OpMode::OPBASE;
    collAlgParams.maxTmpMemSize = 200 * 1024 * 1024;

    CollAlgResReq resReq;
    auto ret = algoExecutor->CalcRes(&virtTopo, resReq);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resReq.primQueueNum, 3);
    std::shared_ptr<InsQueue> insQue(new InsQueue);

    algoExecutor->vTopo_.clear();
    algoExecutor->virtRankMap_.clear();
    algoExecutor->virtRanks_.clear();

    ret = algoExecutor->Orchestrate(&virtTopo, collAlgOp, collAlgParams, insQue);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(insQue->SizeOfSlaves(), 2);
    // check vTpopo size
    EXPECT_EQ(algoExecutor->vTopo_.size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[0].size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[0][0].size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[0][1].size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[1].size(), 1);
    EXPECT_EQ(algoExecutor->vTopo_[1][0].size(), 3);

    // check virtRanks size
    EXPECT_EQ(algoExecutor->virtRanks_.size(), 2);
    EXPECT_EQ(algoExecutor->virtRanks_[0].size(), 4);
    EXPECT_EQ(algoExecutor->virtRanks_[1].size(), 3);

    for (auto iter = insQue->Iter(); iter.HasNext(); ++iter) {
        std::cout << iter->Describe() << std::endl;
    }
}

// {{0,1},{8,9}},{{3,5},{19,21}},对称2D
TEST_F(AllGatherAICPUMesh2dNHRTest, Aicpu_AllGather_Parallel_executor_2x2_2x2)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910_95));

    RankId myRank = 0;
    u32 rankSize = 8;
    VirtualTopoStub virtTopo(myRank);
    string rankTable = "test";
    virtTopo.TopoInit91095TwoPodTwoTwoAndTwoTwo(rankTable);

    std::unique_ptr<InsAllGatherParallelExecutor<TopoMatchConcurrMeshNHR, InsTempAllGatherMesh2D, InsTempAllGatherNHR>>
        algoExecutor(
            new InsAllGatherParallelExecutor<TopoMatchConcurrMeshNHR, InsTempAllGatherMesh2D, InsTempAllGatherNHR>);
    algoExecutor->SetMyRank(myRank);
    algoExecutor->SetRankSize(rankSize);
    algoExecutor->EnableDataAllign(false);
    algoExecutor->EnableDetour(false);
    algoExecutor->SetDevType(DevType::DEV_TYPE_910_95);

    CollAlgOperator collAlgOp;
    collAlgOp.opType = OpType::ALLGATHER;
    collAlgOp.dataCount = 600;
    collAlgOp.dataType = DataType::FP32;
    uint64_t dataSize = collAlgOp.dataCount * 2;

    collAlgOp.inputMem = DevBuffer::Create(0x1000000, dataSize);
    collAlgOp.outputMem = DevBuffer::Create(0x2000000, dataSize * rankSize);
    collAlgOp.scratchMem = DevBuffer::Create(0x3000000, dataSize * rankSize);

    CollAlgParams collAlgParams;
    collAlgParams.opMode = OpMode::OPBASE;
    collAlgParams.maxTmpMemSize = 200 * 1024 * 1024;

    CollAlgResReq resReq;
    auto ret = algoExecutor->CalcRes(&virtTopo, resReq);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resReq.primQueueNum, 3);
    std::shared_ptr<InsQueue> insQue(new InsQueue);

    algoExecutor->vTopo_.clear();
    algoExecutor->virtRankMap_.clear();
    algoExecutor->virtRanks_.clear();

    ret = algoExecutor->Orchestrate(&virtTopo, collAlgOp, collAlgParams, insQue);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(insQue->SizeOfSlaves(), 2);
    // check vTpopo size
    EXPECT_EQ(algoExecutor->vTopo_.size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[0].size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[0][0].size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[0][1].size(), 2);
    EXPECT_EQ(algoExecutor->vTopo_[1].size(), 1);
    EXPECT_EQ(algoExecutor->vTopo_[1][0].size(), 2);

    // check virtRanks size
    EXPECT_EQ(algoExecutor->virtRanks_.size(), 2);
    EXPECT_EQ(algoExecutor->virtRanks_[0].size(), 4);
    EXPECT_EQ(algoExecutor->virtRanks_[1].size(), 2);

    for (auto iter = insQue->Iter(); iter.HasNext(); ++iter) {
        std::cout << iter->Describe() << std::endl;
    }
}