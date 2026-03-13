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
 
class AivAllReduceMesh1DOneShot : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AivAllReduceMesh1DOneShot set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "AivAllReduceMesh1DOneShot tear down" << std::endl;
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
    void RunAivAllReduceMesh1DOneShotTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize) {
 
        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);
 
        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLREDUCE;
        checkerOpParam.tag = "AllReduce";
        checkerOpParam.opMode = opMode;
        checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
        checkerOpParam.algName = algName;
 
        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(AivAllReduceMesh1DOneShot, AivAllReduceMesh1dOneShot_one_four_test)
{
    RunAivAllReduceMesh1DOneShotTest(1, 1, 4, CheckerOpMode::OPBASE, 100, "AivAllReduceMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(AivAllReduceMesh1DOneShot, AivAllReduceMesh1dOneShot_one_three_test)
{
    RunAivAllReduceMesh1DOneShotTest(1, 1, 3, CheckerOpMode::OPBASE, 100, "AivAllReduceMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(AivAllReduceMesh1DOneShot, AivAllReduceMesh1dOneShot_one_eight_test)
{
    RunAivAllReduceMesh1DOneShotTest(1, 1, 8, CheckerOpMode::OPBASE, 100, "AivAllReduceMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(AivAllReduceMesh1DOneShot, AivAllReduceMesh1dOneShot_one_two_test)
{
    RunAivAllReduceMesh1DOneShotTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "AivAllReduceMesh1DOneShot", 1024*1024*200);
}
 
TEST_F(AivAllReduceMesh1DOneShot, AivAllReduceMesh1dOneShot_one_4G_two_test)
{
    RunAivAllReduceMesh1DOneShotTest(1, 1, 2, CheckerOpMode::OPBASE, 100, "AivAllReduceMesh1DOneShot", 1024*1024*20000);
}

TEST_F(AivAllReduceMesh1DOneShot, AivAllReduceMesh1dOneShot_one_eight_4G_test)
{
    RunAivAllReduceMesh1DOneShotTest(1, 1, 8, CheckerOpMode::OPBASE, 100, "AivAllReduceMesh1DOneShot", 1024*1024*20000);
}

TEST_F(AivAllReduceMesh1DOneShot, AivAllReduceMesh1dOneShot_one_six_4G_offload_test_01)
{
    RunAivAllReduceMesh1DOneShotTest(1, 1, 6, CheckerOpMode::OFFLOAD, 100, "InsAllReduceMesh1DOneShot", 1024*1024*10000);
}