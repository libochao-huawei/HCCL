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

class All2AllCCUV2DTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "All2All2D CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "All2All2D CCU test tear down" << std::endl;
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
};

void GenAllToAllVParams(u32 rankSize, u64 count, std::vector<u64>& sendCounts, std::vector<u64>& sdispls,
                        std::vector<u64>& recvCounts, std::vector<u64>& rdispls)
{
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
    return;
}

TEST_F(All2AllCCUV2DTest, all2all2d_ccu_case_test_2_mul_2_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "All2Allv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(268435456, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    GenAllToAllVParams(rankNum, 268435456, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllCCUV2DTest, all2all2d_ccu_case_test_2_mul_2_rank_count1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,8,9}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "All2Allv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(0, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    GenAllToAllVParams(rankNum, 1, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh2D";

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllCCUV2DTest, all2all2d_ccu_case_test_3_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10,16,17,18}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "All2Allv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllCCUV2DTest, all2all2d_ccu_case_test_2_mul_3_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10}}};

    setenv("HCCL_IODIE_NUM", "2", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "All2Allv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllCCUV2DTest, all2all2d_ccu_case_test_2_mul_3_rank_twoloop)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "All2Allv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    GenAllToAllVParams(rankNum, 600000, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllCCUV2DTest, all2all2d_ccu_case_test_2_mul_3_rank_threeloop)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta {{{0,1,2,8,9,10}}};

    setenv("HCCL_IODIE_NUM", "2", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "All2Allv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    GenAllToAllVParams(rankNum, 1200000, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh2D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
}