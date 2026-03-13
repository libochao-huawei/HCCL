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
 
using namespace Hccl;
 
class AivBroadcastMesh1D : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AivBroadcastMesh1D set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "AivBroadcastMesh1D tear down" << std::endl;
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
    void RunAivBroadcastMesh1DTest(int root, int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize) {
 
        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);
 
        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::BROADCAST;
        checkerOpParam.tag = "Broadcast";
        checkerOpParam.opMode = opMode;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
        checkerOpParam.algName = algName;
        checkerOpParam.root = root;
 
        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};
 
TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_four_test)
{
    RunAivBroadcastMesh1DTest(0, 1, 1, 4, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*200);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_four_test_root_1)
{
    RunAivBroadcastMesh1DTest(1, 1, 1, 4, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*200);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_three_test)
{
    RunAivBroadcastMesh1DTest(0, 1, 1, 3, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*200);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_three_test_root_2)
{
    RunAivBroadcastMesh1DTest(2, 1, 1, 3, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*200);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_eight_test)
{
    RunAivBroadcastMesh1DTest(0, 1, 1, 8, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*200);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_two_test)
{
    RunAivBroadcastMesh1DTest(0, 1, 1, 2, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*200);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_4G_two_test)
{
    RunAivBroadcastMesh1DTest(0, 1, 1, 2, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*20000);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_eight_4G_test)
{
    RunAivBroadcastMesh1DTest(0, 1, 1, 8, CheckerOpMode::OPBASE, 100, "AivBroadcastMesh1D", 1024*1024*20000);
}

TEST_F(AivBroadcastMesh1D, AivAllReduceMesh1d_one_six_4G_offload_test_01)
{
    RunAivBroadcastMesh1DTest(0, 1, 1, 6, CheckerOpMode::OFFLOAD, 100, "AivBroadcastMesh1D", 1024*1024*10000);
}