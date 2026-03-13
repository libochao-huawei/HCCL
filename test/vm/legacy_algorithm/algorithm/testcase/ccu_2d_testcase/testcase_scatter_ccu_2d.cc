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

class Scatter2DCCUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Scatter2D CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Scatter2D CCU test tear down" << std::endl;
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

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_2_mul_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_3_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_3_mul_3_rank_512k_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_3_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_4_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_4_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_4_mul_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_5_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,32,33,34,35,36,40,41,42,43,44,48,49,50,51,52}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_2_mul_2_rank_1count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.root = 3;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_8_mul_8_rank_overBuffSize_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = (2 * 1024 * 1024) / sizeof(CheckerDataType::DATA_TYPE_UINT16) + 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT16;
    checkerOpParam.root = 56;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    unsetenv("HCCL_BUFFSIZE");
}

TEST_F(Scatter2DCCUTest, scatter2d_ccu_case_test_5_mul_7_rank_overBuffSize_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,32,33,34,35,36,40,41,42,43,44,48,49,50,51,52}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "Scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = (2 * 1024 * 1024) / sizeof(CheckerDataType::DATA_TYPE_FP8E5M2);
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP8E5M2;
    checkerOpParam.root = 6;
    checkerOpParam.algName = "CcuScatterMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    unsetenv("HCCL_BUFFSIZE");
}


TEST_F(Scatter2DCCUTest, scatter_ccu_case_test_4rank_OPBASE_0count_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT16;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuScatterParallelMesh1DNHR";
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter_ccu_case_test_4rank_OPBASE_4count_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 4;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 1;
    checkerOpParam.algName = "CcuScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter_ccu_case_test_4rank_OPBASE_100count_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.root = 1;
    checkerOpParam.algName = "CcuScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Scatter2DCCUTest, scatter_ccu_mem2mem_opbase_512_int32_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::SCATTER;
    checkerOpParam.tag = "scatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 512;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT64;
    checkerOpParam.root = 1;
    checkerOpParam.algName = "CcuScatterParallelMesh1DNHR";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}



}