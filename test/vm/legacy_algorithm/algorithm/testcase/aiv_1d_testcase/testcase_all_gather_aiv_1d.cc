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
#include "topo_match_mesh_nhr.h"

#include "coll_alg_params.h"
#include "coll_operator.h"
#include "ins_v2_all_gather_sole_executor.h"
#include "ins_all_gather_parallel_executor.h"
#include "ins_coll_alg_base.h"
#include "aiv_temp_all_gather_mesh_1D.h"
#include "topo_match_mesh.h"
#include "ins_temp_all_gather_mesh.h"
#include "ins_temp_all_gather_nhr.h"
#include "topo_match_mesh_nhr.h"
#include "aiv_temp_broadcast_mesh_1D.h"
#include "coll_alg_component.h"
#include "ins_temp_all_gather_mesh.h"
#include "ccu_temp_all_gather_mesh_1D.h"
#undef private


using namespace Hccl;

class AivAllGatherMesh1D : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AivAllGatherMesh1D set up." << std::endl;
    }
 
    static void TearDownTestCase()
    {
        std::cout << "AivAllGatherMesh1D tear down" << std::endl;
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
    void RunAivAllGatherMesh1DTest(int supNum, int sevNum, int rankNum, CheckerOpMode opMode, int dataCount, string algName, int maxTmpMemSize) {

        RankTable_For_LLT gen;
        TopoMeta topoMeta;
        gen.GenTopoMeta(topoMeta, supNum, sevNum, rankNum);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLGATHER;
        checkerOpParam.tag = "AllGather";
        checkerOpParam.opMode = opMode;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
        checkerOpParam.algName = algName;

        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(AivAllGatherMesh1D, AllGather1D_one_four_test)
{
    RunAivAllGatherMesh1DTest(1, 1, 4, CheckerOpMode::OPBASE, 100, "AivAllGatherMesh1D", 1024*1024*200);
}

TEST_F(AivAllGatherMesh1D, AllGather1D_one_three_test)
{
    RunAivAllGatherMesh1DTest(1, 1, 3, CheckerOpMode::OPBASE, 0xDEAD, "AivAllGatherMesh1D", 1024*1024*200);
}

TEST_F(AivAllGatherMesh1D, AllGather1D_one_eight_test)
{
    RunAivAllGatherMesh1DTest(1, 1, 8, CheckerOpMode::OPBASE, 10086, "AivAllGatherMesh1D", 1024*1024*200);
}

TEST_F(AivAllGatherMesh1D, AllGather1D_one_two_test)
{
    RunAivAllGatherMesh1DTest(1, 1, 2, CheckerOpMode::OPBASE, 1314, "AivAllGatherMesh1D", 1024*1024*200);
}

TEST_F(AivAllGatherMesh1D, AllGather1D_one_4G_two_test)
{
    RunAivAllGatherMesh1DTest(1, 1, 2, CheckerOpMode::OPBASE, 918, "AivAllGatherMesh1D", 1024*1024*200);
}


TEST_F(AivAllGatherMesh1D, AllGather1D_one_eight_4G_test)
{
    RunAivAllGatherMesh1DTest(1, 1, 8, CheckerOpMode::OPBASE, 1024*1024*2000, "AivAllGatherMesh1D", 1024);
}

TEST_F(AivAllGatherMesh1D, AllGather_excutor_template_test)
{
    InsV2AllGatherSoleExecutor<TopoMatchMesh,AivTempAllGatherMesh1D> executor;
    std::shared_ptr<InsCollAlgBase> collAlgBase = std::make_shared<InsAllGatherParallelExecutor<TopoMatchMeshNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR>>();
    std::shared_ptr<AivTempBroadcastMesh1D> tempBroadcast = std::make_shared<AivTempBroadcastMesh1D>(
    0, 
    4, 
    std::vector<std::vector<RankId>>{{0, 1, 2, 3}}, 
    std::map<RankId, u32>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
    );
    std::shared_ptr<AivAlgTemplateBase> tempAiv = tempBroadcast;
    u32 numBlocks = 0;
    executor.CalNumBlocks(numBlocks, 1000, 56);
    collAlgBase->CalNumBlocks(numBlocks, 1000, 56);
    tempAiv->CalNumBlocks(numBlocks, 1000, 56);

    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095OneTimesFour(rankTable);
    RankGraph* rankGraphPtr = static_cast<RankGraph*>(&virtTopo);
    CollAlgComponent collAlgComponent(rankGraphPtr, DevType::DEV_TYPE_910_95, u32(0), u32(4));
    u64 dataSize = 1000;
    std::string algName = "AivAllGatherMesh1D";
    collAlgComponent.CalNumBlocks(numBlocks, dataSize, OpType::ALLGATHER, algName, 56);
    std::shared_ptr<InsTempAllGatherMesh1D> tempAllGather = std::make_shared<InsTempAllGatherMesh1D>(
    0, 
    4, 
    std::vector<std::vector<RankId>>{{0, 1, 2, 3}}, 
    std::map<RankId, u32>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
    );
    std::shared_ptr<InsAlgTemplateBase> tempIns = tempAllGather;
    tempIns->CalNumBlocks(numBlocks, 1000, 56);

    std::shared_ptr<CcuTempAllGatherMesh1D> tempAllGatherCcu = std::make_shared<CcuTempAllGatherMesh1D>(
    0, 
    4, 
    std::vector<std::vector<RankId>>{{0, 1, 2, 3}}, 
    std::map<RankId, u32>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
    );
    std::shared_ptr<CcuAlgTemplateBase> tempCcu = tempAllGatherCcu;
    tempCcu->CalNumBlocks(numBlocks, 1000, 56);
}
