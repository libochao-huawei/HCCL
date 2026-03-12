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

class ReduceScatterTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceScatterTest set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "ReduceScatterTest tear down" << std::endl;
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

    void RunReduceScatterTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize) {

        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
        checkerOpParam.tag = "ReduceScatter";
        checkerOpParam.opMode = opMode;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
        checkerOpParam.algName = algName;

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_four_test)
{
    RunReduceScatterTest(1, 1, 4, CheckerOpMode::OPBASE, 100, "InsReduceScatterNHR", 1024*1024*200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_three_test)
{
    RunReduceScatterTest(1, 1, 3, CheckerOpMode::OPBASE, 100, "InsReduceScatterNHR", 1024*1024*200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_two_test)
{
    RunReduceScatterTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "InsReduceScatterNHR", 1024*1024*200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_eight_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OPBASE, 100, "InsReduceScatterNHR", 1024*1024*200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_four_offload_test)
{
    RunReduceScatterTest(1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterNHR", 1024*1024*200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_four_4G_offload_test)
{
    RunReduceScatterTest(1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterNHR", 1024*1024*8000);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_eight_4G_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OPBASE, 100, "InsReduceScatterNHR", 1024*1024*8200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_eight_OFFLOAD_4G_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterNHR", 1024*1024*8200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_eight_OFFLOAD_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterNHR", 1024*1024*200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_eight_offload_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterNHR", 1024*1024*200);
}

TEST_F(ReduceScatterTest, ReduceScatterNHR_one_eight_4G_offload_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterNHR", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_3rank_4G_opbase_test)
{
    RunReduceScatterTest(1, 1, 3, CheckerOpMode::OPBASE, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_6rank_4G_opbase_test)
{
    RunReduceScatterTest(1, 1, 6, CheckerOpMode::OPBASE, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_8rank_4G_opbase_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OPBASE, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_3rank_4G_offload_test)
{
    RunReduceScatterTest(1, 1, 3, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_6rank_4G_offload_test)
{
    RunReduceScatterTest(1, 1, 6, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_8rank_4G_offload_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_2rank_100_offload_test)
{
    RunReduceScatterTest(1, 1, 2, CheckerOpMode::OFFLOAD, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1D_2rank_100_opbase_test)
{
    RunReduceScatterTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "InsReduceScatterMesh1D", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_2rank_big_opbase_test)
{
    RunReduceScatterTest(1, 1, 2, CheckerOpMode::OPBASE, 256 * 1024 * 1024 + 1, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_6rank_100_opbase_test)
{
    RunReduceScatterTest(1, 1, 6, CheckerOpMode::OPBASE, 101, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_8rank_100_opbase_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OPBASE, 101, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_3rank_100_offload_test)
{
    RunReduceScatterTest(1, 1, 3, CheckerOpMode::OFFLOAD, 101, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_6rank_100_offload_test)
{
    RunReduceScatterTest(1, 1, 6, CheckerOpMode::OFFLOAD, 101, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_8rank_100_offload_test)
{
    RunReduceScatterTest(1, 1, 8, CheckerOpMode::OFFLOAD, 101, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_2rank_100_offload_test)
{
    RunReduceScatterTest(1, 1, 2, CheckerOpMode::OFFLOAD, 101, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterMesh1DMeshChunk_2rank_100_opbase_test_)
{
    RunReduceScatterTest(1, 1, 2, CheckerOpMode::OPBASE, 101, "InsReduceScatterMesh1DMeshChunk", 1024*1024*9200);
}

TEST_F(ReduceScatterTest, ReduceScatterParallel_offload_3x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterTest, ReduceScatterParallel_opbase_3x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterTest, ReduceScatterParallel_offload_3x2_big)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 35523320;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterTest, ReduceScatterParallel_opbase_3x2_big)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 35523320;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
 
TEST_F(ReduceScatterTest, ReduceScatterParallel_offload_2x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {0,1}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterTest, ReduceScatterParallel_opbase_2x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {0,1}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduceScatter_aicpu_case_test_ReduceScatterNHR_2pod_2mul2_2mul2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 8, 9}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(ReduceScatterTest, reduceScatter_aicpu_case_test_ReduceScatterNHR_3pod_2mul1_2mul2_3mul1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3}, {8, 9, 10}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}