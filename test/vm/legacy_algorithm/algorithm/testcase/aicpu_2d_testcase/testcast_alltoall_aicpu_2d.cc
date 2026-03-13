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

constexpr u64 K = 1024;
constexpr u64 M = 1024 * K;
constexpr u64 G = 1024 * M;

class All2AllAicpu2DTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "All2All2D Aicpu test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "All2All2D Aicpu test tear down" << std::endl;
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

    std::vector<u64> GenerateDataCount()
    {
        std::set<u64> dataCountSet = {
            1, 2, 4, 8, 16, 128, 1 * K, 2 * K, 256 * K, 512 * K, 1 * M, 200 * M, 256 * M, 500 * M, 1.01 * G};
        return std::vector<u64>(dataCountSet.begin(), dataCountSet.end());
    }

    void RunAlltoAllMesh2DTest(TopoMeta &topoMeta, CheckerOpMode opMode, u64 dataCountPerRank, CheckerDataType sendType)
    {
        RankTable_For_LLT gen;

        setenv("HCCL_IODIE_NUM", "2", 1);
        setenv("HCCL_BUFFSIZE", "200", 1);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLTOALL;
        checkerOpParam.tag = "All2All";
        checkerOpParam.opMode = opMode;

        // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
        u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
        checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(dataCountPerRank, rankNum);
        checkerOpParam.All2AllDataDes.sendType = sendType;
        checkerOpParam.All2AllDataDes.recvType = sendType;
        checkerOpParam.All2AllDataDes.sendCount = dataCountPerRank;

        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.algName = "InsAlltoAllMesh2D";

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(All2AllAicpu2DTest, all2all2d_aicpu_case_test_3_mul_3_rank)
{
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};
    RunAlltoAllMesh2DTest(topoMeta, CheckerOpMode::OPBASE, 1, CheckerDataType::DATA_TYPE_INT8);
}

TEST_F(All2AllAicpu2DTest, all2all2d_aicpu_case_test_3_mul_3_rank_0)
{
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};
    RunAlltoAllMesh2DTest(topoMeta, CheckerOpMode::OPBASE, 0, CheckerDataType::DATA_TYPE_INT8);
}

TEST_F(All2AllAicpu2DTest, all2all2d_aicpu_case_test_4_mul_4_rank)
{
    TopoMeta topoMeta {{{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}}};
    RunAlltoAllMesh2DTest(topoMeta, CheckerOpMode::OPBASE, 67, CheckerDataType::DATA_TYPE_INT8);
}

TEST_F(All2AllAicpu2DTest, all2all2d_aicpu_case_test_5_mul_5_rank)
{
    TopoMeta topoMeta {{{0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,32,33,34,35,36}}};
    RunAlltoAllMesh2DTest(topoMeta, CheckerOpMode::OPBASE, 100, CheckerDataType::DATA_TYPE_FP16);
}

TEST_F(All2AllAicpu2DTest, all2all2d_aicpu_case_test_3_mul_4rank)
{
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26}}};
    RunAlltoAllMesh2DTest(topoMeta, CheckerOpMode::OPBASE, 100, CheckerDataType::DATA_TYPE_INT32);
}

TEST_F(All2AllAicpu2DTest, all2all2d_aicpu_case_test_3_mul_4rank_600_count)
{
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26}}};
    RunAlltoAllMesh2DTest(topoMeta, CheckerOpMode::OPBASE, 600, CheckerDataType::DATA_TYPE_FP32);
}

TEST_F(All2AllAicpu2DTest, all2all2d_aicpu_case_test_3_mul_4rank_600_offload_count_0)
{
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18,24,25,26}}};
    RunAlltoAllMesh2DTest(topoMeta, CheckerOpMode::OFFLOAD, 0, CheckerDataType::DATA_TYPE_INT32);
}

}