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
#include "checker.h"
#include "sim_world.h"
 
class ST_SCATTER_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_test_origin)
{
    int superPodNum = 1;
    int serverNum = 2;
    int rankNum = 8;
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, superPodNum, serverNum, rankNum);
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.DataDes.count = 800;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    opParam.root = 0;
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_test)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 1, 1, 8);
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910B;
    opParam.DataDes.count = 800;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    opParam.root = 0;
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_single_rank_test)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 1, 1, 1);
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910B;
    opParam.DataDes.count = 800;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    opParam.root = 0;
    opParam.algName = "ScatterSingleExecutor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_ScatterCommExecutor_test)
{
    TopoMeta topoMeta {{{2, 5}, {0, 1, 2}}};
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910B;
    opParam.DataDes.count = 100;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    opParam.root = 0;
    opParam.algName = "ScatterCommExecutor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_ScatterMeshExecutor_test)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 1, 2, 8);
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910B;
    opParam.DataDes.count = 100;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    opParam.root = 0;
    opParam.algName = "ScatterMeshExecutor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_ScatterRingExecutor_test)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 1, 2, 8);
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910B;
    opParam.DataDes.count = 100;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    opParam.root = 0;
    opParam.algName = "ScatterRingExecutor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_ScatterRingFor91093Executor_test)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 1, 2, 8);
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.DataDes.count = 100;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    opParam.root = 0;
    opParam.algName = "ScatterRingFor91093Executor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_opbase_ScatterRingFor91093Executor_test_double_pod)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 2, 2, 8);
 
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.DataDes.count = 100;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    opParam.root = 0;
    opParam.algName = "ScatterRingFor91093Executor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_ScatterRingFor91093Executor_NHR)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 2, 2, 8);
 
    setenv("HCCL_ALGO", "level0:NA;level1:NHR;level2:NHR", 1);
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.DataDes.count = 100;
    opParam.root = 0;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.algName = "ScatterRingFor91093Executor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_ScatterRingFor91093Executor_NHR_NSLB)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 2, 2, 3);
 
    setenv("HCCL_ALGO", "level0:NHR;level1:NHR;level2:NHR", 1);
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.DataDes.count = 100;
    opParam.root = 0;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.algName = "ScatterRingFor91093Executor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_ScatterRingFor91093Executor_NB)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 2, 2, 8);
 
    setenv("HCCL_ALGO", "level0:NA;level1:NB;level2:NB", 1);
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.DataDes.count = 100;
    opParam.root = 0;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.algName = "ScatterRingFor91093Executor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_ScatterRingFor91093Executor_NB_NSLB)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 2, 2, 3);
 
    setenv("HCCL_ALGO", "level0:NA;level1:NB;level2:NB", 1);
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.DataDes.count = 100;
    opParam.root = 0;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.algName = "ScatterRingFor91093Executor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}
 
TEST_F(ST_SCATTER_TEST, st_scatter_ScatterRingFor91093Executor_Ring)
{
    TopoMeta topoMeta;
    HcclSim::GenTopoMeta(topoMeta, 2, 2, 8);
 
    setenv("HCCL_ALGO", "level0:NA;level1:ring;level2:ring", 1);
    CheckerOpParam opParam;
    opParam.opType = CheckerOpType::SCATTER;
    opParam.opMode = CheckerOpMode::OPBASE;
    opParam.DataDes.count = 100;
    opParam.root = 0;
    opParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    opParam.devType = CheckerDevType::DEV_TYPE_910_93;
    opParam.algName = "ScatterRingFor91093Executor";
 
    HcclSim::Checker checker;
    EXPECT_TRUE(checker.Check(topoMeta, opParam));
}