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

#include "coll_service_stub.h"
#include "checker.h"
#include "testcase_utils.h"
#include "topo_meta.h"
#define private public
#include "ins_temp_all_reduce_aicpu_reduce.h"
#include "ins_v2_all_reduce_sole_executor.h"
#include "virtual_topo_stub.h"
#include "dev_capability.h"
#include "virtual_topo.h"
#include "orion_adapter_rts.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "ins_coll_alg_base.h"
#include "topo_match_mesh.h"
#include "coll_alg_component.h"
#include "topo_match_nhr.h"
#include "topo_match_concurr_mesh.h"
#undef private
using namespace Hccl;
namespace checker {

constexpr u64 K = 1024;
constexpr u64 M = 1024 * K;
constexpr u64 G = 1024 * M;

std::vector<u64> GenerateDataCount()
{
    std::set<u64> dataCountSet = {
        1, 2, 4, 8, 16, 128, 1 * K, 2 * K, 256 * K, 512 * K, 1 * M, 200 * M, /* 256 * M, 500 * M */};
    for (u64 i = 1; i <= 230 * M; i = (i * 1.3) + 1) {
        dataCountSet.insert(i);
    }
    return std::vector<u64>(dataCountSet.begin(), dataCountSet.end());
}

class AllReduceAICPUMesh1DTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllReduce AICPU 1D test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllReduce AICPU 1D test tear down" << std::endl;
    }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName = "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
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

TEST_F(AllReduceAICPUMesh1DTest, allreduce_aicpu_case_test_2_rank_Mesh1DTwoShot)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.algName = "InsAllReduceMesh1DTwoShot";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1DTest, allreduce_aicpu_case_test_7_rank_Mesh1DTwoShot_smalldata)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 7);

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsAllReduceMesh1DTwoShot";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1DTest, allreduce_aicpu_case_test_4_rank_Mesh1DTwoShot_offload)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "InsAllReduceMesh1DTwoShot";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
TEST_F(AllReduceAICPUMesh1DTest, allreduce_aicpu_case_test_2_rank_cout1024_Mesh1DTwoShot_offload)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsAllReduceMesh1DTwoShot";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1DTest, allreduce_aicpu_case_test_2_rank_count67108864_Mesh1DTwoShot_offload)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 67108864;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsAllReduceMesh1DTwoShot";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1DTest, allreduce_aicpu_reduce_1d_test)
{   
    Hccl::ResLinks resLinks;
    vector<InsQuePtr> queues;
    for (u32 rank = 0; rank < 4; rank++) {
       LinkData link(BasePortType(PortDeploymentType::P2P), 0, rank, 0, 1); 
       resLinks[rank] = {link};
    }
    u32 tempRanksize = 4;
    std::shared_ptr<InsTempAllReduceAicpuReduce> temp = std::make_shared<InsTempAllReduceAicpuReduce>(
    0, 
    tempRanksize, 
    std::vector<std::vector<RankId>>{{0, 1, 2, 3}}, 
    std::map<RankId, u32>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
    );

    InsQuePtr que = std::make_shared<InsQueue>();
    for (u32 i = 0; i < tempRanksize - 1; i++) {
        queues.push_back(que->Fork());
    }

    TempFuncs tempFuncs;
    TemplateDataParams templateData;
    temp->GenExtIns(tempFuncs, templateData, resLinks, queues);
}

}