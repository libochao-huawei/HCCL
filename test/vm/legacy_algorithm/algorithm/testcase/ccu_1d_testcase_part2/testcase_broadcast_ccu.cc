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
#include "mem_layout.h"

namespace checker {

class BroadCastCCUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BroadCast CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BroadCast CCU test tear down" << std::endl;
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
        MemLayout::Global()->Reset();
        Checker::SetDumpFileName("analysis_result");
        GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
public:
    uint32_t rankSize_{0};
};

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_2rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_3rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_3rank_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    rankSize_ = GetRankNumFormTopoMeta(topoMeta);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMesh1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_4rank)
// {
//     RankTable_For_LLT gen;
//     TopoMeta topoMeta;
//     gen.GenTopoMeta(topoMeta, 1, 1, 4);

//     CheckerOpParam checkerOpParam;
//     checkerOpParam.opType = CheckerOpType::BROADCAST;
//     checkerOpParam.tag = "broadcast";
//     checkerOpParam.opMode = CheckerOpMode::OPBASE;
//     checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
//     checkerOpParam.DataDes.count = 100;
//     checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
//     checkerOpParam.algName = "CcuBroadcastMesh1D";
//     checkerOpParam.root = 0;

//     Checker checker;
//     HcclResult ret;
//     ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
//     EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
// }

// TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_4rank_512k_count)
// {
//     RankTable_For_LLT gen;
//     TopoMeta topoMeta;
//     gen.GenTopoMeta(topoMeta, 1, 1, 4);

//     CheckerOpParam checkerOpParam;
//     checkerOpParam.opType = CheckerOpType::BROADCAST;
//     checkerOpParam.tag = "broadcast";
//     checkerOpParam.opMode = CheckerOpMode::OPBASE;
//     checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
//     checkerOpParam.DataDes.count = 512 * 1024;
//     checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
//     checkerOpParam.algName = "CcuBroadcastMesh1D";
//     checkerOpParam.root = 0;

//     Checker checker;
//     HcclResult ret;
//     ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
//     EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
// }

// TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_4rank_offload_512k_count)
// {
//     RankTable_For_LLT gen;
//     TopoMeta topoMeta;
//     gen.GenTopoMeta(topoMeta, 1, 1, 4);

//     rankSize_ = GetRankNumFormTopoMeta(topoMeta);

//     CheckerOpParam checkerOpParam;
//     checkerOpParam.opType = CheckerOpType::BROADCAST;
//     checkerOpParam.tag = "broadcast";
//     checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
//     checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
//     checkerOpParam.DataDes.count = 512 * 1024;
//     checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
//     checkerOpParam.algName = "CcuBroadcastMesh1D";
//     checkerOpParam.root = 0;

//     Checker checker;
//     HcclResult ret;
//     ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
//     EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
// }

// TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_5rank)
// {
//     RankTable_For_LLT gen;
//     TopoMeta topoMeta;
//     gen.GenTopoMeta(topoMeta, 1, 1, 5);

//     rankSize_ = GetRankNumFormTopoMeta(topoMeta);

//     CheckerOpParam checkerOpParam;
//     checkerOpParam.opType = CheckerOpType::BROADCAST;
//     checkerOpParam.tag = "broadcast";
//     checkerOpParam.opMode = CheckerOpMode::OPBASE;
//     checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
//     checkerOpParam.DataDes.count = 100;
//     checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
//     checkerOpParam.algName = "CcuBroadcastMesh1D";
//     checkerOpParam.root = 0;

//     Checker checker;
//     HcclResult ret;
//     ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
//     EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
// }

// TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_6rank)
// {
//     RankTable_For_LLT gen;
//     TopoMeta topoMeta;
//     gen.GenTopoMeta(topoMeta, 1, 1, 6);

//     CheckerOpParam checkerOpParam;
//     checkerOpParam.opType = CheckerOpType::BROADCAST;
//     checkerOpParam.tag = "broadcast";
//     checkerOpParam.opMode = CheckerOpMode::OPBASE;
//     checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
//     checkerOpParam.DataDes.count = 1000;
//     checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
//     checkerOpParam.algName = "CcuBroadcastMesh1D";
//     checkerOpParam.root = 0;

//     Checker checker;
//     HcclResult ret;
//     ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
//     EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
// }

// TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_6rank_MeshMultiMission1D)
// {
//     RankTable_For_LLT gen;
//     TopoMeta topoMeta;
//     gen.GenTopoMeta(topoMeta, 1, 1, 6);

//     CheckerOpParam checkerOpParam;
//     checkerOpParam.opType = CheckerOpType::BROADCAST;
//     checkerOpParam.tag = "broadcast";
//     checkerOpParam.opMode = CheckerOpMode::OPBASE;
//     checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
//     checkerOpParam.DataDes.count = 1000;
//     checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
//     checkerOpParam.algName = "CcuBroadcastMesh1D";
//     checkerOpParam.root = 0;

//     Checker checker;
//     HcclResult ret;
//     ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
//     EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
// }

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_2rank_mem2mem)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_3rank_mem2mem)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_3rank_offload_mem2mem)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_4rank_mem2mem)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_4rank_512k_count_mem2mem)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_4rank_offload_512k_count_mem2mem)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_5rank_mem2mem)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_6rank_mem2mem)
// {
//     RankTable_For_LLT gen;
//     TopoMeta topoMeta;
//     gen.GenTopoMeta(topoMeta, 1, 1, 6);

//     CheckerOpParam checkerOpParam;
//     checkerOpParam.opType = CheckerOpType::BROADCAST;
//     checkerOpParam.tag = "broadcast";
//     checkerOpParam.opMode = CheckerOpMode::OPBASE;
//     checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
//     checkerOpParam.DataDes.count = 1000;
//     checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
//     checkerOpParam.algName = "CcuBroadcastMeshMem2Mem1D";
//     checkerOpParam.root = 0;

//     Checker checker;
//     HcclResult ret;
//     ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
//     EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
// }

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_nhr1d_2rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    rankSize_ = GetRankNumFormTopoMeta(topoMeta);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastNHRMem2Mem1D";
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadCastCCUTest, broadcast_ccu_case_test_nhr1d_3rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    rankSize_ = GetRankNumFormTopoMeta(topoMeta);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "CcuBroadcastNHRMem2Mem1D";
    checkerOpParam.root = 1;

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}