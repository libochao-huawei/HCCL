/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "gtest/gtest.h"

#include <vector>
#include <iostream>
#include <string>

#include "coll_service_stub.h"
#include "checker.h"
#include "testcase_utils.h"
#include "topo_meta.h"

namespace checker{

class AllGatherTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherTest set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherTest tear down" << std::endl;
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

    void RunAllGatherTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, std::string algName, int maxTmpMemSize) {

        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLGATHER;
        checkerOpParam.tag = "AllGather";
        checkerOpParam.opMode = opMode;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
        checkerOpParam.algName = algName;

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(AllGatherTest, AllGatherMesh_one_four_test)
{
    RunAllGatherTest(1, 1, 4, CheckerOpMode::OPBASE, 100, "InsAllGatherMesh", 1024*1024*200);
}

TEST_F(AllGatherTest, AllGatherMesh_one_three_test)
{
    RunAllGatherTest(1, 1, 3, CheckerOpMode::OPBASE, 100, "InsAllGatherMesh", 1024*1024*200);
}

TEST_F(AllGatherTest, AllGatherMesh_one_eight_test)
{
    RunAllGatherTest(1, 1, 8, CheckerOpMode::OPBASE, 100, "InsAllGatherMesh", 1024*1024*400);
}

TEST_F(AllGatherTest, AllGatherMesh_one_two_test)
{
    RunAllGatherTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "InsAllGatherMesh", 1024*1024*200);
}

TEST_F(AllGatherTest, AllGatherNHR_one_four_test)
{
    RunAllGatherTest(1, 1, 4, CheckerOpMode::OPBASE, 100, "InsAllGatherNHR", 1024*1024*200);
}

TEST_F(AllGatherTest, AllGatherNHR_one_three_test)
{
    RunAllGatherTest(1, 1, 3, CheckerOpMode::OPBASE, 100, "InsAllGatherNHR", 1024*1024*200);
}

TEST_F(AllGatherTest, AllGatherNHR_one_eight_test)
{
    RunAllGatherTest(1, 1, 8, CheckerOpMode::OPBASE, 100, "InsAllGatherNHR", 1024*1024*200);
}

TEST_F(AllGatherTest, AllGatherNHR_one_two_test)
{
    RunAllGatherTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "InsAllGatherNHR", 1024*1024*200);
}

TEST_F(AllGatherTest, AllGatherMesh_one_4G_two_test)
{
    RunAllGatherTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "InsAllGatherMesh", 1024*1024*20000);
}

TEST_F(AllGatherTest, AllGatherNHR_one_4G_two_test)
{
    RunAllGatherTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "InsAllGatherNHR", 1024*1024*20000);
}

TEST_F(AllGatherTest, AllGatherMesh_one_4G_offload_two_test)
{
    RunAllGatherTest(1, 1, 2, CheckerOpMode::OFFLOAD, 100, "InsAllGatherMesh", 1024*1024*20000);
}

TEST_F(AllGatherTest, AllGatherNHR_one_4G_offload_two_test)
{
    RunAllGatherTest(1, 1, 2, CheckerOpMode::OFFLOAD, 100, "InsAllGatherNHR", 1024*1024*20000);
}

TEST_F(AllGatherTest, AllGatherNHR_one_to_six_offload_test)
{
    RunAllGatherTest(1, 1, 6, CheckerOpMode::OFFLOAD, 100, "InsAllGatherNHR", 1024*1024*20000);
}

TEST_F(AllGatherTest, AllGatherNHR_one_to_six_test)
{
    RunAllGatherTest(1, 1, 6, CheckerOpMode::OPBASE, 100, "InsAllGatherNHR", 1024*1024*20000);
}

TEST_F(AllGatherTest, AllGatherNHR_one_to_six_01_test)
{
    RunAllGatherTest(1, 1, 6, CheckerOpMode::OPBASE, 400, "InsAllGatherNHR", 1024*1024*20000);
}

TEST_F(AllGatherTest, AllGatherNHR_one_to_six_02_test)
{
    RunAllGatherTest(1, 1, 6, CheckerOpMode::OPBASE, 4000, "InsAllGatherNHR", 1024*1024*20000);
}

}

TEST_F(AllGatherTest, AllGatherParallel_offload_3x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_opbase_3x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_offload_3x2_big)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 35523320;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_opbase_3x2_big)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2}, {0,1,2}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 35523320;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}


TEST_F(AllGatherTest, AllGatherParallel_offload_2x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {0,1}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_opbase_2x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {0,1}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 3556632;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_offload_2x2_little)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {0,2}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 10;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_offload_3pods)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 10;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_offload_4pods)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_asymmetric_offload_2n4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 10;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_nhr_offload_2n3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {1, 2, 3}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 10;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_nhr_offload_3pods)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3}, {5, 6, 7}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_asymmetric_4n6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}, {0, 1, 2, 8, 9, 10}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_asymmetric_offload_6n6n9)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10}, {3,4,5,11,12,13},{4,5,6,12,13,14,20,21,22}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh1DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_nhr_offload_6n7)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, AllGatherParallel_nhr_4n5)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}, {0, 1, 2, 3, 4}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}