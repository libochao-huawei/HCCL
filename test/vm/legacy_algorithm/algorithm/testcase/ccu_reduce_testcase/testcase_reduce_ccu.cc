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

class ReduceCCUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Reduce CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Reduce CCU test tear down" << std::endl;
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

TEST_F(ReduceCCUTest, reduce_ccu_case_test_2rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_case_test_3rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_case_test_5rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_case_test_5rank_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_case_test_5rank_offload_512k_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_case_test_3rank_Mesh1DMultiMission)
{
    // 此算法用例有死锁问题
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_m2m_case_test_5rank_offload_512k_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_m2m_case_test_5rank_offload_0_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_m2m_case_test_3rank_opbase_100_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 1;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_m2m_case_test_2rank_opbase_1024_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.root = 1;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_m2m_case_test_8rank_opbase_1025_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.DataDes.count = 1025;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.root = 3;
    checkerOpParam.algName = "CcuReduceMeshMem2Mem1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceCCUTest, reduce_ccu_nhr_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "CcuReduceNHR1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}  // namespace checker