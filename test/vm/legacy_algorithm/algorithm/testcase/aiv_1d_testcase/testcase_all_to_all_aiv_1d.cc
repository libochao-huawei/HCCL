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
#include "topo_match_mesh.h"
#include "aiv_temp_all_to_all_mesh_1D.h"
#include "ins_v2_all_to_all_sole_executor.h"
#include "ins_coll_alg_base.h"

using namespace Hccl;

class AivAlltoAllMesh1D : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AivAlltoAllMesh1D set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AivAlltoAllMesh1D tear down" << std::endl;
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

    void RunAivAlltoAllMesh1DTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCountPerRank,
        CheckerDataType dataType, int maxTmpMemSize) {

        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLTOALL;
        checkerOpParam.tag = "All2All";
        checkerOpParam.opMode = opMode;

        // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
        checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(dataCountPerRank, rankNum);
        checkerOpParam.All2AllDataDes.sendType = dataType;
        checkerOpParam.All2AllDataDes.recvType = dataType;
        checkerOpParam.All2AllDataDes.sendCount = dataCountPerRank;

        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.algName = "AivAlltoAllMesh1D";

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_one_two_test)
{
    RunAivAlltoAllMesh1DTest(1, 1, 2, CheckerOpMode::OPBASE, 100, CheckerDataType::DATA_TYPE_INT32, 1024*1024*200);
}

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_one_four_test)
{
    RunAivAlltoAllMesh1DTest(1, 1, 4, CheckerOpMode::OPBASE, 100, CheckerDataType::DATA_TYPE_INT32, 1024*1024*200);
}

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_one_three_test)
{
    RunAivAlltoAllMesh1DTest(1, 1, 3, CheckerOpMode::OPBASE, 100, CheckerDataType::DATA_TYPE_INT32, 1024*1024*200);
}

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_one_eight_test)
{
    RunAivAlltoAllMesh1DTest(1, 1, 8, CheckerOpMode::OPBASE, 100, CheckerDataType::DATA_TYPE_INT32, 1024*1024*200);
}

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_one_4G_two_test)
{
    RunAivAlltoAllMesh1DTest(1, 1, 2, CheckerOpMode::OPBASE, 4*1024*1024*1024,
        CheckerDataType::DATA_TYPE_INT32, 1024*1024*200);
}

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_one_eight_4G_test)
{
    RunAivAlltoAllMesh1DTest(1, 1, 8, CheckerOpMode::OPBASE, 4*1024*1024*1024,
        CheckerDataType::DATA_TYPE_INT32, 1024*1024*200);
}

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_one_six_4G_offload)
{
    RunAivAlltoAllMesh1DTest(1, 1, 6, CheckerOpMode::OFFLOAD, 100, CheckerDataType::DATA_TYPE_INT32, 1024*1024*200);
}

TEST_F(AivAlltoAllMesh1D, AivAlltoAllMesh1d_calculate_numblocks)
{
    RankId myRank = 0;
    u32 rankSize = 4;
    std::vector<std::vector<RankId>> tempVTopo = {{0, 1, 2, 3}};
    std::map<RankId, u32> tempVirtRankMap = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    auto executor = std::make_shared<InsV2AlltoAllSoleExecutor<TopoMatchMesh, AivTempAlltoAllMesh1D>>();
    executor->SetRankSize(rankSize);
    auto temp = std::make_shared<AivTempAlltoAllMesh1D>(myRank, rankSize, tempVTopo, tempVirtRankMap);

    u32 numBlocks = 0;
    HcclResult ret = HcclResult::HCCL_SUCCESS;

    ret = executor->CalNumBlocks(numBlocks, 1000, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 1);
    ret = temp->CalNumBlocks(numBlocks, 1000, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 1);

    ret = executor->CalNumBlocks(numBlocks, 1000, 3);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 2);
    ret = temp->CalNumBlocks(numBlocks, 1000, 3);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 2);

    ret = executor->CalNumBlocks(numBlocks, 1000, 11);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 8);
    ret = temp->CalNumBlocks(numBlocks, 1000, 11);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 8);

    ret = executor->CalNumBlocks(numBlocks, 1000, 48);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 48);
    ret = temp->CalNumBlocks(numBlocks, 1000, 48);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(numBlocks, 48);
}