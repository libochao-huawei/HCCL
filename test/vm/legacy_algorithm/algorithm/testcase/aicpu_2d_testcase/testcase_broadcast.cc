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

    void RunBroadcastTest2D(int root, TopoMeta &topoMeta, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize)
    {   
        setenv("HCCL_IODIE_NUM", "2", 1);
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

TEST_F(BroadcastTest, BroadcastMesh2D_two_two_test)
{   
    TopoMeta topoMeta {{{0,1,8,9}}};
    RunBroadcastTest2D(0, topoMeta, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_two_two_test_root_1)
{   
    TopoMeta topoMeta {{{0,1,8,9}}};
    RunBroadcastTest2D(1, topoMeta, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_three_four_test)
{   
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19}}};
    RunBroadcastTest2D(0, topoMeta, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_three_four_test_root_6)
{   
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19}}};
    RunBroadcastTest2D(6, topoMeta, CheckerOpMode::OPBASE, 100, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_two_two_test_count_1)
{   
    TopoMeta topoMeta {{{0,1,8,9}}};
    RunBroadcastTest2D(0, topoMeta, CheckerOpMode::OPBASE, 1, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_two_two_test_count_0)
{   
    TopoMeta topoMeta {{{0,1,8,9}}};
    RunBroadcastTest2D(0, topoMeta, CheckerOpMode::OPBASE, 1, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_two_two_test_count_datacount0)
{   
    TopoMeta topoMeta {{{0,1,8,9}}};
    RunBroadcastTest2D(0, topoMeta, CheckerOpMode::OPBASE, 0, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_two_two_test_count_OFFLOAD_0)
{   
    TopoMeta topoMeta {{{0,1,8,9}}};
    RunBroadcastTest2D(0, topoMeta, CheckerOpMode::OFFLOAD, 1, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

TEST_F(BroadcastTest, BroadcastMesh2D_two_two_test_count_OFFLOAD_datacount0)
{   
    TopoMeta topoMeta {{{0,1,8,9}}};
    RunBroadcastTest2D(0, topoMeta, CheckerOpMode::OFFLOAD, 0, "InsBroadcastMesh2DTwoShot", 1024*1024*200);
}

}