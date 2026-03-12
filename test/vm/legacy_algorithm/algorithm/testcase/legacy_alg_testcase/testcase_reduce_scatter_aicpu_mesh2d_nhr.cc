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

class ReduceScatterAICPUMesh2dNHRTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceScatter AICPU ParrallelMesh2DNHR test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceScatter AICPU ParrallelMesh2DNHR test tear down" << std::endl;
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


TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_2_pod_ParallelMesh2DNHR)
{
    TopoMeta topoMeta{{{0, 1, 8, 9, 24, 25, 32, 33}, {3, 5, 19, 21}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 400;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_2_server_ParallelMesh2DNHR)
{
    TopoMeta topoMeta{{{0, 1, 8, 9}, {16, 17, 24, 25}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 400;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_3_pod_ParallelMesh2DNHR)
{
    TopoMeta topoMeta{{{0, 1, 8, 9, 16, 17}, {16, 17, 24, 25, 32, 33}, {0, 1, 8, 9, 16, 17}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 400;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_3_pod2mul2_ParallelMesh2DNHR)
{
    TopoMeta topoMeta{{{0, 1, 8, 9}, {16, 17, 24, 25}, {0, 1, 8, 9}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 400;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_2_server_3mul3_ParallelMesh2DNHR)
{
    TopoMeta topoMeta{{{0, 1, 2, 8, 9, 10, 16, 17, 18}, {16, 17, 18, 24, 25, 26, 32, 33, 34}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 400;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_2_server_3mul2_ParallelMesh2DNHR)
{
    TopoMeta topoMeta{{{0, 1, 8, 9, 16, 17}, {16, 17, 24, 25, 32, 33}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 400;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_2_pod_Symmetric_2mul2)
{
    TopoMeta topoMeta{{{0, 1, 8, 9}, {16, 17, 24, 25}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 / sizeof(DATA_TYPE_UINT16);
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT16;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    unsetenv("HCCL_BUFFSIZE");
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_2_pod_Symmetric_4mul4)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27},
                       {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT8;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_3_pod_Symmetric_4mul3)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19},
                       {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19},
                       {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT64;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_3_pod_Symmetric_3mul3)
{
    TopoMeta topoMeta{{{0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26},
                       {4, 5, 6, 12, 13, 14, 20, 21, 22, 28, 29, 30},
                       {0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1025;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_2_pod_Asymmetric_2mul2_2mul4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}, {2, 3, 10, 11, 18, 19, 26, 27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_3_pod_Asymmetric_2mul2_2mul4_2mul6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}, {2, 3, 10, 11, 18, 19, 26, 27}, {4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterAICPUMesh2dNHRTest, reduceScatter_aicpu_case_test_3_pod_Asymmetric_2mul2_2mul4_2mul4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}, {2, 3, 10, 11, 18, 19, 26, 27}, {4, 5, 12, 13, 20, 21, 28, 29}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "InsReduceScatterParallelMesh2DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

