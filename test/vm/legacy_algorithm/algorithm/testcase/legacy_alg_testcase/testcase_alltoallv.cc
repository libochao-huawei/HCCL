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

using namespace Hccl;

class AllToAllVTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllToAllVTest set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "AllToAllVTest tear down" << std::endl;
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

    void GenAllToAllVParams(u32 rankSize, u64 count, std::vector<u64>& sendCounts, std::vector<u64>& sdispls,
                            std::vector<u64>& recvCounts, std::vector<u64>& rdispls) {
        u64 sendDisplacement = 0;
        u64 recvDisplacement = 0;
        for (u32 i = 0; i < rankSize; i++) {
            sendCounts.push_back(count);
            sdispls.push_back(sendDisplacement);
            recvCounts.push_back(count);
            rdispls.push_back(recvDisplacement);
            sendDisplacement += count;
            recvDisplacement += count;
        }
    }

    void RunAlltoAllvMeshTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize) {
        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLTOALLV;
        checkerOpParam.tag = "AllToAllV";
        checkerOpParam.opMode = opMode;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
        checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;
        checkerOpParam.algName = algName;

        GenAllToAllVParams(rankNum, dataCount, checkerOpParam.All2AllDataDes.sendCounts, 
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);
        
        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(AllToAllVTest, AlltoAllVMesh_one_four_test)
{
    RunAlltoAllvMeshTest(1, 1, 4, CheckerOpMode::OPBASE, 100, "InsAlltoAllvMesh", 1024*1024*200);
}

TEST_F(AllToAllVTest, AlltoAllVMesh_one_four_test_01)
{
    RunAlltoAllvMeshTest(1, 1, 4, CheckerOpMode::OPBASE, 200, "InsAlltoAllvMesh", 1024*1024*200);
}

TEST_F(AllToAllVTest, AlltoAllVMesh_one_three_test)
{
    RunAlltoAllvMeshTest(1, 1, 3, CheckerOpMode::OPBASE, 100, "InsAlltoAllvMesh", 1024*1024*200);
}

TEST_F(AllToAllVTest, AlltoAllVMesh_one_fix_test)
{
    RunAlltoAllvMeshTest(1, 1, 6, CheckerOpMode::OPBASE, 100, "InsAlltoAllvMesh", 1024*1024*200);
}

TEST_F(AllToAllVTest, AlltoAllVMesh_one_four_offload_test)
{
    RunAlltoAllvMeshTest(1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsAlltoAllvMesh", 1024*1024*200);
}

TEST_F(AllToAllVTest, AlltoAllVMesh_one_four_4G_offload_test)
{
    RunAlltoAllvMeshTest(1, 1, 4, CheckerOpMode::OFFLOAD, 100, "InsAlltoAllvMesh", 1024*1024*6000);
}

TEST_F(AllToAllVTest, AlltoAllVMesh_one_four_4G_test)
{
    RunAlltoAllvMeshTest(1, 1, 4, CheckerOpMode::OPBASE, 100, "InsAlltoAllvMesh", 1024*1024*6000);
}