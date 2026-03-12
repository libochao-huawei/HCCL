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

using namespace Hccl;

class AllReduceTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllReduceTest set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllReduceTest tear down" << std::endl;
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
    void RunAllReduceTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, CheckerReduceOp reduceType,
        CheckerDataType dataType, int dataCount, string algName, int maxTmpMemSize) {

        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLREDUCE;
        checkerOpParam.tag = "AllReduce";
        checkerOpParam.opMode = opMode;
        checkerOpParam.reduceType = reduceType;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = dataType;
        checkerOpParam.algName = algName;

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

// nhr
TEST_F(AllReduceTest, AllReduceNHR_one_four_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_three_test)
{
    RunAllReduceTest(1, 1, 3, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_eight_test)
{
    RunAllReduceTest(1, 1, 8, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_two_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_4G_two_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*20000);
}

TEST_F(AllReduceTest, AllReduceNHR_one_4G_offload_two_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*20000);
}

TEST_F(AllReduceTest, AllReduceNHR_one_eight_4G_test)
{
    RunAllReduceTest(1, 1, 8, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*20000);
}

TEST_F(AllReduceTest, AllReduceNHR_one_six_offload_test)
{
    RunAllReduceTest(1, 1, 6, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_six_4G_offload_test)
{
    RunAllReduceTest(1, 1, 6, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*10000);
}

TEST_F(AllReduceTest, AllReduceNHR_one_six_4G_offload_test_01)
{
    RunAllReduceTest(1, 1, 6, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceNHR", 1024*1024*10000);
}

// 补充nhr用例
TEST_F(AllReduceTest, AllReduceNHR_one_four_zero_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        0, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_four_small_data_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        3, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_three_min_test)
{
    RunAllReduceTest(1, 1, 3, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_MIN, CheckerDataType::DATA_TYPE_INT32,
        2000, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_eight_odd_max_test)
{
    RunAllReduceTest(1, 1, 8, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_MAX, CheckerDataType::DATA_TYPE_BFP16,
        1999, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_two_big_data_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        1024*1024*512, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_two_big_non_divisible_data_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_MIN, CheckerDataType::DATA_TYPE_INT32,
        1024*1024*513, "InsAllReduceNHR", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceNHR_one_two_big_non_divisible_data_offload_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_MAX, CheckerDataType::DATA_TYPE_FP16,
        1024*1024*513, "InsAllReduceNHR", 1024*1024*200);
}

// mesh 1D one shot
TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_four_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_three_test)
{
    RunAllReduceTest(1, 1, 3, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_eight_test)
{
    RunAllReduceTest(1, 1, 8, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_two_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_4G_two_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*20000);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_4G_offload_two_test)
{
    RunAllReduceTest(1, 1, 2, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*20000);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_eight_4G_test)
{
    RunAllReduceTest(1, 1, 8, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*20000);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_six_offload_test)
{
    RunAllReduceTest(1, 1, 6, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*200);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_six_4G_offload_test)
{
    RunAllReduceTest(1, 1, 6, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        100, "InsAllReduceMesh1DOneShot", 1024*1024*10000);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_four_1G_opbase_count17_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_MIN, CheckerDataType::DATA_TYPE_FP16,
        17, "InsAllReduceMesh1DOneShot", 1024*1024);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_four_1G_opbase_count18_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_BFP16,
        18, "InsAllReduceMesh1DOneShot", 1024*1024);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_four_1G_opbase_count550001_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        550001*200, "InsAllReduceMesh1DOneShot", 1024*1024);
}

TEST_F(AllReduceTest, AllReduceMesh1DOneShot_one_four_1G_opbase_count550000_test)
{
    RunAllReduceTest(1, 1, 4, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, CheckerDataType::DATA_TYPE_INT32,
        550000*200, "InsAllReduceMesh1DOneShot", 1024*1024);
}

TEST_F(AllReduceTest, allreduce_aicpu_case_test_AllReduceNHR_2pod_2mul2_2mul2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 8, 9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllReduceNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aicpu_case_test_AllReduceNHR_3pod_2mul1_2mul2_3mul1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3}, {8, 9, 10}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllReduceNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}