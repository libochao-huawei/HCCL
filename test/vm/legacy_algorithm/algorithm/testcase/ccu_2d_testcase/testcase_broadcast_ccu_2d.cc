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
#include "ccu_component.h"

namespace checker {

class BroadCast2DCCUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BroadCast2D CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BroadCast2D CCU test tear down" << std::endl;
    }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName = "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
        Checker::SetDumpFileName(caseName);
    }

    virtual void TearDown()
    {
        for (uint32_t idx = 0; idx < rankSize_; idx++) {
            Hccl::CcuComponent::GetInstance(idx).Deinit();
        }
        Checker::SetDumpFileName("analysis_result");
        GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }

public:
    uint32_t rankSize_{0};
};

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_4_mul_4_rank_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1024", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 256 * 1024 * 1024 / sizeof(DATA_TYPE_INT8) + 1024;  // ub(256M)
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 3;
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_2_uint8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 3;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT8;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_3rank_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_3rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_3_rank_512k_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_3_rank_offload_512k_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_5_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26,32,33,34}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26,32,33,34,40,41,42,48,49,50}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_4_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,40,41,42,43,48,49,50,51}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_5_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,32,33,34,35,36,40,41,42,43,44,48,49,50,51,52}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_6_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,8,9,10,11,12,13,16,17,18,19,20,21,24,25,26,27,28,29,32,33,34,35,36,37,40,41,42,43,44,45,48,49,50,51,52,53}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_4_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_5_mul_5_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,32,33,34,35,36}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_6_mul_6_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,8,9,10,11,12,13,16,17,18,19,20,21,24,25,26,27,28,29,32,33,34,35,36,37,40,41,42,43,44,45}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_2_rank_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    setenv("HCCL_BUFFSIZE", "1024", 1);  // buffsize --> 1M
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1024 * 1024 * 1024 / sizeof(s32) - 1;  // 1G - 1
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 2;
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_2_rank_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 / sizeof(s64);  // 2M
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 3;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_2_rank_3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT8;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_8_rank_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT16;
    checkerOpParam.DataDes.count = 1024 * 1024 / sizeof(u16) - 1;  // 1M - 1
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_8_rank_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT32;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 / sizeof(u32);  // 2M
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_2_mul_8_rank_3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_8_mul_8_rank_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
        24,25,26,27,28,29,30,31,
        32,33,34,35,36,37,38,39,
        40,41,42,43,44,45,46,47,
        48,49,50,51,52,53,54,55,
        56,57,58,59,60,61,62,63}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT16;
    checkerOpParam.DataDes.count = 1024 * 1024 / sizeof(u16) - 1;  // 1M - 1
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_8_mul_8_rank_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
        24,25,26,27,28,29,30,31,
        32,33,34,35,36,37,38,39,
        40,41,42,43,44,45,46,47,
        48,49,50,51,52,53,54,55,
        56,57,58,59,60,61,62,63}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 / sizeof(u32);  // 2M
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 63;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_8_mul_8_rank_3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16,17,18,19,20,21,22,23,
        24,25,26,27,28,29,30,31,
        32,33,34,35,36,37,38,39,
        40,41,42,43,44,45,46,47,
        48,49,50,51,52,53,54,55,
        56,57,58,59,60,61,62,63}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT8;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 25;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_mem2mem_8_mul_8_rank_4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
        24,25,26,27,28,29,30,31,
        32,33,34,35,36,37,38,39,
        40,41,42,43,44,45,46,47,
        48,49,50,51,52,53,54,55,
        56,57,58,59,60,61,62,63}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1024", 1);
    constexpr u64 GB2B = 1024 * 1024 * 1024;
    constexpr u64 HUGE_DATA_SIZE = 2 * GB2B;

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = HUGE_DATA_SIZE / sizeof(u64) + 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_UINT64;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 16;
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_4_rank_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1024 * 1024 / sizeof(u16) - 1;  // 1M - 1
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 3;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_4_rank_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 / sizeof(u32);  // 2M
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 6;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_3_mul_3_rank_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    constexpr u64 MB2B = 1024 * 1024;
    constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * MB2B;

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = MAX_OFFLOAD_SCRATCH_SIZE / sizeof(DATA_TYPE_FP16) + 1024 + 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_4_mul_4_rank_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    rankSize_ = GetRankNumFormTopoMeta(topoMeta);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_ccu_case_test_5_mul_7_rank_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,27,28,29,30,32,33,34,35,36,37,38}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 200 * 1024 * 1024 + 1;  // 200M
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem2D";
    checkerOpParam.root = 17;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// multimission 用例
TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_2_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 200 * 1024 + 55;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_2_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_2_mul_3rank_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9,16,17}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_3_mul_3rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_3_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_3_mul_5_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26,32,33,34}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_4_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,40,41,42,43,48,49,50,51}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_5_mul_7_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,32,33,34,35,36,40,41,42,43,44,48,49,50,51,52}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_4_mul_4_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast2d_multimission_ccu_case_test_5_mul_5_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,32,33,34,35,36}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024 + 33;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh2D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast_ccu_mem2mem_int16_small_opbase_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 200;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.algName = "CcuBroadcastParallelMesh1DNHR";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast_ccu_mem2mem_int8_big_opbase_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "CcuBroadcastParallelMesh1DNHR";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast_ccu_mem2mem_fp32_small_opbase_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 200;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastParallelMesh1DNHR";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast_ccu_mem2mem_int32_big_offload_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 50 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "CcuBroadcastParallelMesh1DNHR";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast_ccu_mem2mem_int64_small_offload_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 200;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "CcuBroadcastParallelMesh1DNHR";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCast2DCCUTest, broadcast_ccu_mem2mem_fp32_zero_offload_parallel)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1}, {2,3}}};
    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastParallelMesh1DNHR";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}