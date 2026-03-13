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

class ReduceScatterVCCUTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Scatter CCU test set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Scatter CCU test tear down" << std::endl;
    }

    virtual void SetUp()
    {
        const ::testing::TestInfo *const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName =
            "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
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

TEST_F(ReduceScatterVCCUTest, reduce_scatter_v_ccu_case_test_2rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    vector<u64> counts{100, 200};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "CcuReduceScatterVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVCCUTest, reduce_scatter_v_ccu_case_test_2rank_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    vector<u64> counts{300, 200};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "CcuReduceScatterVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVCCUTest, reduce_scatter_v_ccu_case_test_4rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);
    vector<u64> counts{300, 0, 500, 200};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "CcuReduceScatterVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVCCUTest, reduce_scatter_v_ccu_case_test_8rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    vector<u64> counts{0, 300, 500, 1024, 300*1024, 0, 500, 0};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "CcuReduceScatterVMesh1D";

    Checker checker;
    HcclResult ret;
    ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVCCUTest, reduce_scatter_v_ccu_case_test_4rank_auto_test)
{
    vector<u64> randomNumbersList {
        106, 41, 38, 126, 122, 33, 25, 13, 20, 3, 86, 110, 53, 9, 23, 104, 75, 71, 27, 55, 103, 42, 10, 100, 45,
        74, 29, 82, 65, 107, 120, 12, 116, 59, 92, 64, 31, 18, 22, 98, 95, 84, 124, 7, 115, 117, 8, 66, 34, 16,
        111, 4, 78, 46, 2, 49, 67, 81, 24, 114, 76, 21, 36, 52, 113, 50, 70, 15, 108, 39, 97, 60, 91, 43, 1, 93,
        101, 119, 109, 51, 6, 61, 112, 121, 127, 17, 62, 88, 26, 35, 105, 63, 118, 19, 54, 80, 48, 96, 128, 11
    };

    vector<u64> randomNumbersMultipiler {
        1, 0, 6, 4, 1024, 3, 10, 7, 2, 8, 9, 1, 2, 1024 * 256, 0, 512, 3, 7
    };

    vector<CheckerDataType> dataTypeList {
        CheckerDataType::DATA_TYPE_BFP16, CheckerDataType::DATA_TYPE_INT16,
        CheckerDataType::DATA_TYPE_INT32, CheckerDataType::DATA_TYPE_FP16,
        CheckerDataType::DATA_TYPE_FP32,
    };

    vector<CheckerOpMode> opModeList{CheckerOpMode::OFFLOAD, CheckerOpMode::OPBASE};

    vector<CheckerReduceOp> reduceTypeList{
        CheckerReduceOp::REDUCE_SUM, CheckerReduceOp::REDUCE_MAX, CheckerReduceOp::REDUCE_MIN};

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    u32 rankSize = 4;
    gen.GenTopoMeta(topoMeta, 1, 1, rankSize);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuReduceScatterVMesh1D";

    u64 randomNumbersIdx = 0;
    u64 multipilerIterIdx = 0;
    u64 dataTypeIdx = 0;
    u64 opModeIdx = 0;
    u64 reduceTypeIdx = 0;

    u64 loopNum = 20;

    for (auto i = 1; i < loopNum; ++i) {
        std::cout << "CcuReduceScatterV auto test " << i << " start" << std::endl;
        checkerOpParam.opMode = opModeList[opModeIdx++ % opModeList.size()];
        checkerOpParam.reduceType = reduceTypeList[reduceTypeIdx++ % reduceTypeList.size()];
        checkerOpParam.VDataDes.dataType = dataTypeList[dataTypeIdx++ % dataTypeList.size()];
        checkerOpParam.VDataDes.counts.clear();
        checkerOpParam.VDataDes.displs = {0};
        for (auto r = 0; r < rankSize; ++r) {
            u64 count = randomNumbersMultipiler[multipilerIterIdx++ % randomNumbersMultipiler.size()] *
                        randomNumbersList[randomNumbersIdx++ % randomNumbersList.size()];
            checkerOpParam.VDataDes.counts.push_back(count);
            if (r != 0) {
                checkerOpParam.VDataDes.displs.push_back(
                    checkerOpParam.VDataDes.displs[r - 1] + checkerOpParam.VDataDes.counts[r - 1]);
            }
        }
        std::cout << "counts: ";
        for (auto &x : checkerOpParam.VDataDes.counts) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << "displs: ";
        for (auto &x : checkerOpParam.VDataDes.displs) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(ReduceScatterVCCUTest, reduce_scatter_v_mem2mem_ccu_case_test_4rank_auto_test)
{
    vector<u64> randomNumbersList {
        106, 41, 38, 126, 122, 33, 25, 13, 20, 3, 86, 110, 53, 9, 23, 104, 75, 71, 27, 55, 103, 42, 10, 100, 45,
        74, 29, 82, 65, 107, 120, 12, 116, 59, 92, 64, 31, 18, 22, 98, 95, 84, 124, 7, 115, 117, 8, 66, 34, 16,
        111, 4, 78, 46, 2, 49, 67, 81, 24, 114, 76, 21, 36, 52, 113, 50, 70, 15, 108, 39, 97, 60, 91, 43, 1, 93,
        101, 119, 109, 51, 6, 61, 112, 121, 127, 17, 62, 88, 26, 35, 105, 63, 118, 19, 54, 80, 48, 96, 128, 11
    };

    vector<u64> randomNumbersMultipiler {
        1, 0, 6, 4, 1024, 3, 10, 7, 2, 8, 9, 1, 2, 1024 * 256, 0, 512, 3, 7
    };

    vector<CheckerDataType> dataTypeList {
        CheckerDataType::DATA_TYPE_BFP16, CheckerDataType::DATA_TYPE_INT16,
        CheckerDataType::DATA_TYPE_INT32, CheckerDataType::DATA_TYPE_FP16,
        CheckerDataType::DATA_TYPE_FP32,
    };

    vector<CheckerOpMode> opModeList{CheckerOpMode::OFFLOAD, CheckerOpMode::OPBASE};

    vector<CheckerReduceOp> reduceTypeList{
        CheckerReduceOp::REDUCE_SUM, CheckerReduceOp::REDUCE_MAX, CheckerReduceOp::REDUCE_MIN};

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    u32 rankSize = 4;
    gen.GenTopoMeta(topoMeta, 1, 1, rankSize);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_95;
    checkerOpParam.algName = "CcuReduceScatterVMeshMem2Mem1D";

    u64 randomNumbersIdx = 0;
    u64 multipilerIterIdx = 0;
    u64 dataTypeIdx = 0;
    u64 opModeIdx = 0;
    u64 reduceTypeIdx = 0;

    u64 loopNum = 20;

    for (auto i = 1; i < loopNum; ++i) {
        std::cout << "CcuReduceScatterV auto test " << i << " start" << std::endl;
        checkerOpParam.opMode = opModeList[opModeIdx++ % opModeList.size()];
        checkerOpParam.reduceType = reduceTypeList[reduceTypeIdx++ % reduceTypeList.size()];
        checkerOpParam.VDataDes.dataType = dataTypeList[dataTypeIdx++ % dataTypeList.size()];
        checkerOpParam.VDataDes.counts.clear();
        checkerOpParam.VDataDes.displs = {0};
        for (auto r = 0; r < rankSize; ++r) {
            u64 count = randomNumbersMultipiler[multipilerIterIdx++ % randomNumbersMultipiler.size()] *
                        randomNumbersList[randomNumbersIdx++ % randomNumbersList.size()];
            checkerOpParam.VDataDes.counts.push_back(count);
            if (r != 0) {
                checkerOpParam.VDataDes.displs.push_back(
                    checkerOpParam.VDataDes.displs[r - 1] + checkerOpParam.VDataDes.counts[r - 1]);
            }
        }
        std::cout << "counts: ";
        for (auto &x : checkerOpParam.VDataDes.counts) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << "displs: ";
        for (auto &x : checkerOpParam.VDataDes.displs) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        Checker checker;
        HcclResult ret;
        ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
}
}