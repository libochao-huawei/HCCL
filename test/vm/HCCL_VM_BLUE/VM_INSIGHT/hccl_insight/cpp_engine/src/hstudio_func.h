/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HSTUDIO_FUNC_H
#define HSTUDIO_FUNC_H

#include "checker.h"

using namespace checker;

const int HSTUDIO_OK = 0;
const int HSTUDIO_FAIL = 1;

// 命令参数索引编号，命令格式 ./open_hccl_test -run ./parameter_files AllToAllTest.alltoall_test_910B_opbase
const int PARA_INDEX_PROGRAM = 0;
const int PARA_INDEX_CMDTYPE = 1;
const int PARA_INDEX_CFGPATH = 2;
const int PARA_INDEX_CASENAME = 3;
const int PARA_INDEX_MAXNUM = 4;

struct AlgCaseParaS {
    // for TopoMeta
    bool isAsymmetricTopo { false }; // 网络配置方式是否非对称，默认对称
    struct {
        int superPodNum{1};
        int serverNum{1};
        int rankNum{1};
    } symmetricTopo;
    struct {
        TopoMeta super_server_ranks;  // std::vector<std::vector<std::vector<int>>>
    } asymmetricTopo;

    CheckerOpType opType { CheckerOpType::ALLREDUCE };
    std::string algName;
    CheckerOpMode opMode { OPBASE };
    CheckerReduceOp reduceType { CheckerReduceOp::REDUCE_RESERVED };
    CheckerDevType devtype { CheckerDevType::DEV_TYPE_910B };
    bool is310P3V = false;  // 仅当310PV卡的时候，设置为1
    u32 root { INVALID_VALUE_RANKID  };
    u32 dstRank { 1 };
    u32 srcRank { 0 };
    u64 count { 160 };
    CheckerDataType dataType { CheckerDataType::DATA_TYPE_FP32 };
    std::map<std::string, std::string> envVars;

    void Print() const {
        std::cout  << ":: topoMode =" << (isAsymmetricTopo ? "Asymmetric" : "Symmetric")
        << ", opType =" << static_cast<int>(opType) << ", opMode =" << static_cast<int>(opMode) << ", reduceType =" << static_cast<int>(reduceType)
        << ", devtype =" << static_cast<int>(devtype) << ", algName =" << algName
        << ", count =" << count << ", dataType =" << static_cast<int>(dataType) << std::endl;
    }
};

struct AlgCaseInfo {
    std::string name_; /* 完整用例名，SuiteName.CaseName */
    AlgCaseParaS parameter_;  /* testcase里调被测接口需要的参数，对应Studio工具的option面板 */
};

bool GetAllAlgTestCases(std::vector<std::string>& names);
HcclResult ExecAlgTestCases(std::string name);
bool GetAlgTestCaseInfo(std::string name,  AlgCaseInfo& caseInfo);
bool CreateAlgTestCase(std::string name,  AlgCaseParaS& para);
bool InitialAlgTest(std::string cfgpath, std::string filter = "");
int ProcStudioCmd(std::string cmd, std::string casename = "");
int HStudioMain(int argc, char **argv);

#endif