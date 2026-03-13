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

class Reduce2DCCUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Reduce2D CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Reduce2D CCU test tear down" << std::endl;
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

TEST_F(Reduce2DCCUTest, reduce2d_ccu_case_test_2_mul_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_ccu_case_test_3_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 8, 9, 10, 16, 17, 18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_ccu_case_test_3_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_ccu_case_test_4_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_ccu_case_test_4_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_ccu_case_test_4_mul_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 8, 9, 10, 11}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_ccu_case_test_5_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0,1,2,3,4,8,9,
                        10,11,12,16,17,18,19,
                        20,24,25,26,27,28,32,
                        33,34,35,36,40,41,42,
                        43,44,48,49,50,51,52}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_m2m_ccu_case_test_2_mul_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 8, 9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_m2m_ccu_case_test_5_mul_5_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.root = 1;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_m2m_ccu_case_test_4_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 1025;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 2;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce2d_m2m_ccu_case_test_5_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0,1,2,3,4,8,9,
                        10,11,12,16,17,18,19,
                        20,24,25,26,27,28,32,
                        33,34,35,36,40,41,42,
                        43,44,48,49,50,51,52}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.root = 5;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce_ccu_mem2mem_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 200;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "CcuReduceParallelMesh1DNHR";
    checkerOpParam.root = 1;
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce_ccu_mem2mem_parallel_root2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.algName = "CcuReduceParallelMesh1DNHR";
    checkerOpParam.root = 0;
 
    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce_ccu_mem2mem_parallel_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 10000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "CcuReduceParallelMesh1DNHR";
    checkerOpParam.root = 0;
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(Reduce2DCCUTest, reduce_ccu_mem2mem_parallel_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "CcuReduceParallelMesh1DNHR";
    checkerOpParam.root = 0;
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}  // namespace checker