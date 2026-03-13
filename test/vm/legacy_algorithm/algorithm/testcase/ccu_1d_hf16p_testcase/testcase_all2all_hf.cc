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

namespace checker {

class All2AllCCUHFTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "All2All CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "All2All CCU test tear down" << std::endl;
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

TEST_F(All2AllCCUHFTest, all2all_ccu_case_test_2Die)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    setenv("HCCL_IODIE_NUM", "2", 1);
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "All2All";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(600, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
     checkerOpParam.All2AllDataDes.sendCount = 600;


    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAllToAllMesh1D2Die";

    Checker checker;
    HcclResult ret;

    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    // 待调试
    // EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllCCUHFTest, all2all_ccu_case_test_2_2Die)
{
    RankTable_For_LLT gen;

    setenv("HCCL_IODIE_NUM", "2", 1);
    TopoMeta topoMeta {{{0,1,8,9}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "All2All";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAllToAllMesh1D2Die";

    Checker checker;
    HcclResult ret;

    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    // 待调试
    // EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

}

TEST_F(All2AllCCUHFTest, all2all_ccu_case_test_3_2Die)
{
    RankTable_For_LLT gen;
    setenv("HCCL_IODIE_NUM", "2", 1);
    TopoMeta topoMeta {{{0,1,2,8,9,10}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "All2All";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(30, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
     checkerOpParam.All2AllDataDes.sendCount = 30;


    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAllToAllMesh1D2Die";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    // 待调试
    // EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllCCUHFTest, all2all_ccu_case_test_4_2Die)
{
    RankTable_For_LLT gen;
    setenv("HCCL_IODIE_NUM", "2", 1);
    TopoMeta topoMeta {{{1,2,3,4,9,10,11,12}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "All2All";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(0, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
     checkerOpParam.All2AllDataDes.sendCount = 0;


    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAllToAllMesh1D2Die";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    // 待调试
    // EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}
