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

namespace checker {

constexpr u64 K = 1024;
constexpr u64 M = 1024 * K;
constexpr u64 G = 1024 * M;

std::vector<u64> GenDataCount()
{
    std::set<u64> dataCountSet = {
        1, 2, 4, 8, 16, 128, 1 * K, 2 * K, 256 * K, 512 * K, 1 * M, 200 * M, 230 * M, /* 256 * M, 500 * M */};
    return std::vector<u64>(dataCountSet.begin(), dataCountSet.end());
}

class AllReduceAICPUMesh1dNHRTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllReduce AICPU ParrallelMesh1DNHR test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllReduce AICPU ParrallelMesh1DNHR test tear down" << std::endl;
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

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_1_rank_ParallelMesh1DNHR)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0},{0}}};

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
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
   
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_2_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1},{0,1}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";
    
    Checker checker;
    auto ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_2_rank_ParallelMesh1DNHR_smalldata)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1},{0,1}}};

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
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}


TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_3_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2},{0,1,2}}};

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
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_3_mul_3_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2},{0,1,2},{0,1,2}}};

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
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_4_rank_ParallelMesh1DNHR_bigdata)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3},{0,1,2,3}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 210 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_8_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7},{0,1,2,3,4,5,6,7}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_2_rank_ParallelMesh1DNHR_0)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1},{0,1}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_3_mul_3_rank_ParallelMesh1DNHR_offload)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2},{0,1,2},{0,1,2}}};

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
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_mul_2_rank_ParallelMesh1DNHR_offload_0)
{
    // 此算法有ERROR级别日志报错
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1},{0,1}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    // buffersize: 200 * 1024 * 1024
    setenv("HCCL_BUFFSIZE", "200", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_Asymmetric_2pod_2mul1_2mul2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_Asymmetric_2pod_2mul2_3mul2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}, {0, 1, 2, 8, 9, 10}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.DataDes.count = 200;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, allreduce_aicpu_case_test_2_pod_Asymmetric_3mul3_3mul2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 8, 9, 10, 16, 17, 18}, {2, 3, 4, 10, 11, 12}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceAICPUMesh1dNHRTest, AllGatherParallel_asymmetric_opbase_6n6n9)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 8, 9, 10}, {3, 4, 5, 11, 12, 13}, {4, 5, 6, 12, 13, 14, 20, 21, 22}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllReduceParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
}