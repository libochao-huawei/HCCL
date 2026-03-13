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

class ScatterTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterTest set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "ScatterTest tear down" << std::endl;
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

    void RunScatterTest(int root, int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize) {

        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::SCATTER;
        checkerOpParam.tag = "scatter";
        checkerOpParam.opMode = opMode;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
        checkerOpParam.root = root;
        checkerOpParam.algName = algName;

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(ScatterTest, ScatterNHR_one_four_test)
{
    RunScatterTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsScatterNHR", 1024*1024*200);
}

TEST_F(ScatterTest, ScatterNHR_one_three_test)
{
    RunScatterTest(0, 1, 1, 3, CheckerOpMode::OPBASE, 100, "InsScatterNHR", 1024*1024*200);
}

TEST_F(ScatterTest, ScatterNHR_one_two_test)
{
    RunScatterTest(0, 1, 1, 2, CheckerOpMode::OPBASE, 100, "InsScatterNHR", 1024*1024*200);
}

TEST_F(ScatterTest, ScatterNHR_one_eight_test)
{
    RunScatterTest(0, 1, 1, 8, CheckerOpMode::OPBASE, 100, "InsScatterNHR", 1024*1024*200);
}

TEST_F(ScatterTest, ScatterNHR_one_four_offload_test)
{
    RunScatterTest(0, 1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsScatterNHR", 1024*1024*200);
}

TEST_F(ScatterTest, ScatterNHR_one_four_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsScatterNHR", 1024*1024*8000);
}

TEST_F(ScatterTest, ScatterNHR_one_eight_4G_test)
{
    RunScatterTest(0, 1, 1, 8, CheckerOpMode::OPBASE, 100, "InsScatterNHR", 1024*1024*8200);
}

TEST_F(ScatterTest, ScatterNHR_one_eight_OFFLOAD_4G_test)
{
    RunScatterTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsScatterNHR", 1024*1024*8200);
}

TEST_F(ScatterTest, ScatterNHR_one_eight_OFFLOAD_test)
{
    RunScatterTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsScatterNHR", 1024*1024*200);
}

TEST_F(ScatterTest, ScatterNHR_one_eight_offload_test)
{
    RunScatterTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsScatterNHR", 1024*1024*200);
}

TEST_F(ScatterTest, ScatterNHR_one_eight_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsScatterNHR", 1024*1024*9200);
}

TEST_F(ScatterTest, ScatterMesh1D_3rank_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 3, CheckerOpMode::OFFLOAD, 100, "InsScatterMesh1D", 1024*1024*9200);
}

TEST_F(ScatterTest, ScatterMesh1D_3rank_4G_test)
{
    RunScatterTest(0, 1, 1, 3, CheckerOpMode::OPBASE, 100, "InsScatterMesh1D", 1024*1024*9200);
}

TEST_F(ScatterTest, ScatterMesh1D_6rank_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 6, CheckerOpMode::OFFLOAD, 100, "InsScatterMesh1D", 1024*1024*9200);
}

TEST_F(ScatterTest, ScatterMesh1D_7rank_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 7, CheckerOpMode::OFFLOAD, 100, "InsScatterMesh1D", 1024*1024*9200);
}

TEST_F(ScatterTest, ScatterMesh1D_5rank_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 5, CheckerOpMode::OFFLOAD, 100, "InsScatterMesh1D", 1024*1024*9200);
}

TEST_F(ScatterTest, ScatterMesh1D_4rank_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsScatterMesh1D", 1024*1024*9200);
}

TEST_F(ScatterTest, ScatterMesh1D_8rank_4G_offload_test)
{
    RunScatterTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsScatterMesh1D", 1024*1024*9200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x2_small)
{   
    RunScatterTest(0, 1, 2, 2, CheckerOpMode::OPBASE, 100, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x8_small)
{   
    RunScatterTest(0, 1, 2, 8, CheckerOpMode::OPBASE, 1000, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_3x8_small)
{   
    RunScatterTest(0, 1, 3, 8, CheckerOpMode::OPBASE, 1000, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x8_big)
{   
    RunScatterTest(0, 1, 2, 8, CheckerOpMode::OPBASE, 100000000, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x8_offload)
{   
    RunScatterTest(0, 1, 2, 8, CheckerOpMode::OFFLOAD, 1000, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x8_root_1)
{   
    RunScatterTest(1, 1, 2, 8, CheckerOpMode::OPBASE, 100, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x8_root_2)
{   
    RunScatterTest(2, 1, 2, 8, CheckerOpMode::OPBASE, 100, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x8_root_9)
{   
    RunScatterTest(9, 1, 2, 8, CheckerOpMode::OPBASE, 100, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_opbase_2x8_count_0)
{   
    RunScatterTest(0, 1, 2, 8, CheckerOpMode::OPBASE, 0, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}

TEST_F(ScatterTest, InsScatterParallelMesh1DNHR_offload_2x8_big)
{   
    RunScatterTest(15, 1, 2, 8, CheckerOpMode::OFFLOAD, 100000000, "InsScatterParallelMesh1DNHR", 1024*1024*200);
}
