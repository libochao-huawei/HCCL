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
 
GTEST_API_ int main(int argc, char **argv) {
    // testcase调试代码，只跑特定的用例
    //testing::GTEST_FLAG(filter) = "AivAllGatherMesh1D.AllGather_excutor_template_test";
    testing::InitGoogleTest(&argc, argv);
    // 用来控制各个LLT任务的topoPath.json的名字，不能和其他任务的名字一样，可能会冲突
    setenv("TOPO_PATH_NAME", "aiv_1d_testcase", 1);
    return RUN_ALL_TESTS();
}