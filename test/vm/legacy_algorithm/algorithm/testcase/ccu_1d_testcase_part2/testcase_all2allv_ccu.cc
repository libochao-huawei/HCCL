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
#include "rank_info_recorder.h"

namespace checker {

class All2AllVCCUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "All2AllV CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "All2AllV CCU test tear down" << std::endl;
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
        RankInfoRecorder::Global()->Reset();
        Checker::SetDumpFileName("analysis_result");
        GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
public:
    uint32_t rankSize_{0};
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

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_2rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_3rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_4rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}


TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_7rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 7);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank_1000_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 1000, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank_offload_200_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 200, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank_Mesh1D)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuHalfAll2AllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL); // TODO
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank_Mesh1D_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuHalfAll2AllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL); // TODO
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_3rank_Mesh1D_opbase_overBuffSize)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_UINT16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_UINT16;
    u64 count = (10 * 1024 * 1024) / sizeof(CheckerDataType::DATA_TYPE_UINT16) + 1;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    unsetenv("HCCL_BUFFSIZE");
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank_Mesh1D_opbase_overBuffSize)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP8E4M3;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP8E4M3;
    u64 count = (10 * 1024 * 1024) / sizeof(CheckerDataType::DATA_TYPE_FP8E4M3);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    unsetenv("HCCL_BUFFSIZE");
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_5rank_Mesh1D_offload_overBuffSize)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    rankSize_ = GetRankNumFormTopoMeta(topoMeta);

    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_HIF8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_HIF8;
    u64 count = (10 * 1024 * 1024) / sizeof(CheckerDataType::DATA_TYPE_HIF8) - 1;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    unsetenv("HCCL_BUFFSIZE");
}

TEST_F(All2AllVCCUTest, all2allv_ccu_case_test_8rank_Mesh1D_offload_overBuffSize)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    setenv("HCCL_BUFFSIZE", "1", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP32;
    u64 count = (2 * 1024 * 1024) / sizeof(CheckerDataType::DATA_TYPE_FP32);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);

    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuAlltoAllVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    unsetenv("HCCL_BUFFSIZE");
}
}