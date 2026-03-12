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

#include "types.h"
#define private public
#include "testcase_utils.h"
#include "virtual_topo_stub.h"
#include "dev_capability.h"
#include "virtual_topo.h"
#include "orion_adapter_rts.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "ins_v2_all_to_all_v_sole_executor.h"
#include "ins_coll_alg_base.h"
#include "aiv_temp_all_to_all_v_mesh_1D.h"
#include "topo_match_mesh.h"
#include "coll_alg_component.h"
#undef private

using namespace Hccl;

class AivAlltoAllVMesh1D : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AivAlltoAllVMesh1D set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AivAlltoAllVMesh1D tear down" << std::endl;
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

    void GenAivAllToAllVParams(u32 rankSize, u64 count, std::vector<u64>& sendCounts, std::vector<u64>& sdispls,
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

    void RunAivAlltoAllVMesh1DTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount,
        CheckerDataType dataType, int maxTmpMemSize) {

        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLTOALLV;
        checkerOpParam.tag = "AllToAllV";
        checkerOpParam.opMode = opMode;

        checkerOpParam.All2AllDataDes.sendType = dataType;
        checkerOpParam.All2AllDataDes.recvType = dataType;
        checkerOpParam.All2AllDataDes.sendCount = dataCount;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.algName = "AivAlltoAllVMesh1D";

        GenAivAllToAllVParams(rankNum, dataCount, checkerOpParam.All2AllDataDes.sendCounts,
            checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts,
            checkerOpParam.All2AllDataDes.rdispls);

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_1)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 4, CheckerOpMode::OPBASE, 0, CheckerDataType::DATA_TYPE_INT8, 1024*1024);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_2)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 4, CheckerOpMode::OFFLOAD, 1, CheckerDataType::DATA_TYPE_INT16, 1024*1024);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_3)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 3, CheckerOpMode::OPBASE, 13, CheckerDataType::DATA_TYPE_INT32, 1024*1024);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_4)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 3, CheckerOpMode::OFFLOAD, 99, CheckerDataType::DATA_TYPE_INT64, 1024*1024);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_5)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 8, CheckerOpMode::OPBASE, 4096, CheckerDataType::DATA_TYPE_UINT8, 1024*1024);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_6)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 2, CheckerOpMode::OFFLOAD, 128*1024*1024, CheckerDataType::DATA_TYPE_UINT16, 1024*1024);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_7)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 2, CheckerOpMode::OPBASE, 1024*1024*1024, CheckerDataType::DATA_TYPE_UINT32, 1024*1024);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_8)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 8, CheckerOpMode::OFFLOAD, 4*1024*1024*1024, CheckerDataType::DATA_TYPE_UINT64, 1024*1024*200);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_9)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 6, CheckerOpMode::OPBASE, 10*1024*1024*1024, CheckerDataType::DATA_TYPE_FP16, 1024*1024*200);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_10)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 6, CheckerOpMode::OFFLOAD, 1024*1024*1024, CheckerDataType::DATA_TYPE_FP32, 1024*1024*200);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_test_11)
{
    RunAivAlltoAllVMesh1DTest(1, 1, 6, CheckerOpMode::OPBASE, 1024*1024*1024, CheckerDataType::DATA_TYPE_BFP16, 1024*1024*200);
}

TEST_F(AivAlltoAllVMesh1D, AivAlltoAllVMesh1d_excutor_template_test)
{
    InsV2AlltoAllVSoleExecutor<TopoMatchMesh, AivTempAlltoAllVMesh1D> executor;
    std::shared_ptr<AivTempAlltoAllVMesh1D> tempAlltoAllV = std::make_shared<AivTempAlltoAllVMesh1D>(
    0,
    4,
    std::vector<std::vector<RankId>>{{0, 1, 2, 3}},
    std::map<RankId, u32>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
    );
    std::shared_ptr<AivAlgTemplateBase> tempAiv = tempAlltoAllV;
    AlgTopoInfo topoInfo;
    topoInfo.virtRanks = std::vector<std::vector<RankId>>{{0, 1, 2, 3}};
    topoInfo.virtRankMap = std::vector<std::map<RankId, u32>>{{{0, 0}, {1, 1}, {2, 2}, {3, 3}}};
    topoInfo.vTopo = std::vector<std::vector<std::vector<RankId>>>{{{{0, 1, 2, 3}}}};
    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095OneTimesFour(rankTable);
    RankGraph* rankGraphPtr = static_cast<RankGraph*>(&virtTopo);
    CollAlgOperator op;
    op.opType = OpType::ALLTOALLV;
    op.all2AllVDataDes.sendType = Hccl::DataType::INT8;
    op.all2AllVDataDes.recvType = Hccl::DataType::INT8;

    u64 sendCounts[4] = {100, 100, 100, 100};
    u64 sdispls[4] = {0, 100, 200, 300};
    u64 recvCounts[4] = {100, 100, 100, 100};
    u64 rdispls[4] = {0, 100, 200, 300};
    op.all2AllVDataDes.sendCounts = (void*)sendCounts;
    op.all2AllVDataDes.sdispls = (void*)sdispls;
    op.all2AllVDataDes.recvCounts = (void*)recvCounts;
    op.all2AllVDataDes.rdispls = (void*)rdispls;

    CollAlgParams params;
    params.maxTmpMemSize = 1000;
    ConnectedLinkMgr linkMgr{};
    CollAlgResReq algResReq;
    CollOffloadOpResReq resReq;
    u64 dataSize = 1000;
    Hccl::DataSlice usrInSlice = Hccl::DataSlice(Hccl::BufferType::INPUT, 0, dataSize);
    Hccl::DataSlice usrOutSlice = Hccl::DataSlice(Hccl::BufferType::OUTPUT, 0, dataSize);
    std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
    InsQuePtr queue = std::make_shared<InsQueue>();
    queue->Append(std::move(insLocalCopy));

    executor.SetMyRank(0);
    executor.SetRankSize(4);
    executor.Orchestrate(topoInfo, op, params, &linkMgr, queue);
    executor.CalcRes(rankGraphPtr, algResReq);
    executor.CalcResOffload(rankGraphPtr, dataSize, resReq);
}