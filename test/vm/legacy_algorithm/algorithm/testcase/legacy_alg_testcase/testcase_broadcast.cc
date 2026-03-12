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

namespace checker{

class BroadcastTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BroadcastTest set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "BroadcastTest tear down" << std::endl;
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

    void RunBroadcastTest(int root, int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize)
    {
        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::BROADCAST;
        checkerOpParam.tag = "broadcast";
        checkerOpParam.opMode = opMode;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
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

TEST_F(BroadcastTest, BroadcastMeshOneShot_one_four_test_01)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DOneShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMeshOneShot_one_four_test)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastMeshOneShot_one_eight_test)
{
    RunBroadcastTest(1, 1, 1, 8, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastMeshOneShot_one_four_test_offload)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsBroadcastMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastMeshOneShot_one_four_test_offload_1)
{
    RunBroadcastTest(1, 1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsBroadcastMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastMeshOneShot_one_four_test_bigdata)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 500000, "InsBroadcastMesh1DOneShot", 1024*1024*1);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_four_test)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_four_4G_test)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_five_4G_test)
{
    RunBroadcastTest(0, 1, 1, 5, CheckerOpMode::OPBASE, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_six_4G_test)
{
    RunBroadcastTest(0, 1, 1, 6, CheckerOpMode::OPBASE, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_seven_4G_test)
{
    RunBroadcastTest(0, 1, 1, 7, CheckerOpMode::OPBASE, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_three_test)
{
    RunBroadcastTest(0, 1, 1, 3, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_two_test)
{
    RunBroadcastTest(0, 1, 1, 2, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_test)
{
    RunBroadcastTest(0, 1, 1, 8, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_four_offload_test)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_four_4G_offload_test)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OFFLOAD, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*8000);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_4G_test)
{
    RunBroadcastTest(0, 1, 1, 8, CheckerOpMode::OPBASE, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_OFFLOAD_4G_test)
{
    RunBroadcastTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_OFFLOAD_test)
{
    RunBroadcastTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_offload_test)
{
    RunBroadcastTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_4G_offload_test)
{
    RunBroadcastTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_4G_opbase_test_1)
{
    RunBroadcastTest(1, 1, 1, 8, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_4G_offload_test_2)
{
    RunBroadcastTest(2, 1, 1, 8, CheckerOpMode::OFFLOAD, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadCastMesh1DTwoShot_one_eight_4G_offload_test_3)
{
    RunBroadcastTest(3, 1, 1, 8, CheckerOpMode::OFFLOAD, 1024*1024*1024*4, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastMesh1DTwoShot_one_four_test_bigdata)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 550000, "InsBroadcastMesh1DTwoShot", 1024*1024*1);

}

TEST_F(BroadcastTest, BroadcastMesh1DTwoShot_one_four_test_02)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DOneShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh1DTwoShot_one_four_test_03)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh1DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastNHR_one_four_test_01)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsBroadcastNHR", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastNHR_one_four_test)
{
    RunBroadcastTest(1, 1, 1, 4, CheckerOpMode::OPBASE, 100, "InsBroadcastNHR", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastNHR_one_eight_test)
{
    RunBroadcastTest(0, 1, 1, 8, CheckerOpMode::OFFLOAD, 100, "InsBroadcastNHR", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastNHR_one_four_test_offload)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OFFLOAD, 500000, "InsBroadcastNHR", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastNHR_one_four_test_offload_1)
{
    RunBroadcastTest(1, 1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsBroadcastNHR", 1024*1024*200);
}
 
TEST_F(BroadcastTest, BroadcastNHR_one_four_test_bigdata)
{
    RunBroadcastTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 500000, "InsBroadcastNHR", 1024*1024*1);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_server_two_two_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_server_two_eight_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_server_eight_eight_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.root = 0;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_diagonal_root_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.root = 31;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_random_root_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.root = 11;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_one_count_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.root = 5;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_bigdata_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 2 * 200 * 1024 * 1024 + 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.root = 11;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, BroadcastMesh1DNHR_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "Broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.root = 3;
    checkerOpParam.algName = "InsBroadcastParallelMesh1DNHR";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}