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

class AllGather2DTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGather 2D test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGather 2D test tear down" << std::endl;
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

TEST_F(AllGather2DTest, AllGatherParallel_opbase_2x2_small)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}, {0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh2DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGather2DTest, AllGatherParallel_opbase_4n8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}, {0,1,8,9,16,17,24,25}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh2DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGather2DTest, AllGatherParallel_opbase_4n8_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}, {2,3,10,11,18,19,26,27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh2DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGather2DTest, AllGatherParallel_opbase_4n8n12)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}, {2,3,10,11,18,19,26,27},{4,5,12,13,20,21,28,29,36,37,44,45}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh2DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGather2DTest, AllGatherParallel_opbase_4n8n8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}, {2,3,10,11,18,19,26,27},{4,5,12,13,20,21,28,29}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh2DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGather2DTest, AllGatherParallel_opbase_6n6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10}, {4,5,6,12,13,14}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsAllGatherParallelMesh2DNHR";

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}