/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "cmd_utils.h"
#include "hccl_vm_log.h"
#include "cmd_base.h"

int main(int argc, char *argv[])
{
    LogConfig config;
    config.consoleLevel = 0;
    InitLogger(config);

    const char* envCheck = std::getenv(HVM_BASH_ENV_KEY.c_str());

    if (envCheck != nullptr) {
        StartHostClient(argc, argv);
    } else {
        std::string cmd = ArgvToString(argc, argv);
        if (argc == 1) {
            cmd += " --help";
        }
        ParseCommand(cmd);
    }
    return 0;
}