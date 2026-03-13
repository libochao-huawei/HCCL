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
 
namespace checker {
 
class ReduceScatterAicpu2DTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceScatter Aicpu 2D test set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "ReduceScatter Aicpu 2D test tear down" << std::endl;
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
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_count_100)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_count_1024)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_count_0)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_count_OFFLOAD_0)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_count_2051)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 2051;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_big_data_200mb)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 21 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_big_data_52428800)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 5242880;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_big_data_26214402)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 2621440;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_big_data_33554432)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 3355443;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_big_data_33554435)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 3355443;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_2_rank_small_data)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_3_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_4_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 9;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_3_rank_small_data)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 20 * 1024 * 1024 + 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_3_rank_count_1024)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_3_rank_count_2051)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 2051;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_3_rank_big_data_200mb)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 20 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_3_rank_big_data_52428800)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 5242880;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterAicpu2DTest, reducescatter_aicpu_case_test_2_mul_3_rank_big_data_26214402)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "200", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 2621440;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterMesh2D";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
}