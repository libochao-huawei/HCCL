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

class All2AllVCCUHFTest : public testing::Test {
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

TEST_F(All2AllVCCUHFTest, all2allv_ccu_Mesh2Die_case_test_2x2rank_opbase_smallSize)
{
    TopoMeta topoMeta {{{0,2, 8,10}}};    // 2x2
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
 
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP32;
    u64 count = (64 * 1024) / sizeof(float32_t);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);
 
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAlltoAllVMesh2Die";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
 
    unsetenv("HCCL_BUFFSIZE");
}
 
TEST_F(All2AllVCCUHFTest, all2allv_ccu_Mesh2Die_case_test_2x3rank_offload_overMs)
{
    TopoMeta topoMeta {{{0,1,2, 8,9,10}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
 
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_UINT16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_UINT16;
    u64 count = (2 * 1024 * 1024) / sizeof(uint16_t);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);
 
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAlltoAllVMesh2Die";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
 
TEST_F(All2AllVCCUHFTest, all2allv_ccu_Mesh2Die_case_test_2x6rank_opbase_smallSize)
{
    TopoMeta topoMeta {{{0,1,2,4,5,6, 8,9,10,12,13,14}}};
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
 
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP64;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP64;
    u64 count = 10 / sizeof(float64_t);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);
 
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAlltoAllVMesh2Die";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
 
    unsetenv("HCCL_BUFFSIZE");
}
 
TEST_F(All2AllVCCUHFTest, all2allv_ccu_Mesh2Die_case_test_16rank_offload_overMs)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);
 
    setenv("HCCL_IODIE_NUM", "2", 1);
 
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
 
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
    u64 count = (360 * 1024) / sizeof(int32_t);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(rankNum,
        count,
        checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);
 
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_HF;
    checkerOpParam.algName = "CcuAlltoAllVMesh2Die";
 
    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

}
