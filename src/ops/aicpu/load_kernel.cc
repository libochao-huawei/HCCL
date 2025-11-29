/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_kernel.h"
#include "mmpa_api.h"
#include "log.h"
#include "adapter_acl.h"
namespace ops_hccl {

aclrtBinHandle g_binKernelHandle = nullptr;

HcclResult ParseLibraryPath(std::string &cannPath)
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_LD_LIBRARY_PATH, mmSysGetEnvValue);
    std::string getPath = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (getPath == "EmptyString") {
        HCCL_ERROR("[ParseLibraryPath]ENV:LD_LIBRARY_PATH is not set");
        return HCCL_E_PARA;
    } else {
        cannPath = getPath;
    }
    return HCCL_SUCCESS;
}

HcclResult GetKeyWordPath(const std::string &cannEnvStr, const std::string &keyStr, std::string &cannPath)
{
    std::string tempPath;   // 存放临时路径
    // 查找cann安装路径
    for (u32 i = 0; i < cannEnvStr.length(); ++i) {
        // 环境变量中存放的每段路径之间以':'隔开
        if (cannEnvStr[i] != ':') {
            tempPath += cannEnvStr[i];
        }

        if (cannEnvStr[i] == ':' || i == cannEnvStr.length() - 1) {
            size_t found = tempPath.find(keyStr);
            if (found == std::string::npos) {
                tempPath.clear();
                continue;
            }
            if (tempPath.length() <= found + keyStr.length() || tempPath[found + keyStr.length()] == '/') {
                cannPath = tempPath.substr(0, found + keyStr.length());
                break;
            }
            tempPath.clear();
        }
    }
    if (cannPath.empty()) {
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult LoadBinaryFromFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
                              aclrtBinHandle &binHandle, bool isRelativePath)
{
    CHK_PRT_RET(binPath == nullptr,
        HCCL_ERROR("[Load][Binary]binary path is nullptr"),
        HCCL_E_PTR);

    std::string cannPath(binPath); // 存放cann安装路径
    // TODO: 需要配合打包路径进行修改
    // if (isRelativePath) {
    //     std::string libraryPath;
    //     HcclResult ret = ParseLibraryPath(libraryPath);
    //     CHK_PRT_RET(ret != HCCL_SUCCESS,
    //         HCCL_ERROR("[LoadBinaryFromFile]errNo[0x%016llx]parse path fail.", ret), ret);

    //     ret = GetKeyWordPath(libraryPath, "/hccl", cannPath);
    //     CHK_PRT_RET(ret != HCCL_SUCCESS,
    //         HCCL_ERROR("[LoadBinaryFromFile]cannot found version file in %s.", libraryPath.c_str()),
    //         HCCL_E_PARA);
    //     cannPath += binPath;
    // }

    printf("[%s:%d]cannPath is %s\n", __FUNCTION__, __LINE__, cannPath.c_str());

    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(realpath(cannPath.c_str(), realPath) == nullptr,
        HCCL_ERROR("LoadBinaryFromFile: %s is not a valid real path, err[%d]", cannPath.c_str(), errno),
        HCCL_E_INTERNAL);
    HCCL_INFO("[LoadBinaryFromFile]realPath: %s", realPath);

    aclrtBinaryLoadOptions loadOptions = {0};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = optionType;
    option.value.cpuKernelMode = cpuKernelMode;

    printf("[%s:%d]realPath is %s\n", __FUNCTION__, __LINE__, realPath);
    aclError aclRet = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle); // ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE
    CHK_PRT_RET(aclRet != ACL_SUCCESS,
        HCCL_ERROR("[LoadBinaryFromFile]errNo[0x%016llx] load binary from file error.", aclRet),
        HCCL_E_OPEN_FILE_FAILURE);

    return HCCL_SUCCESS;
}

HcclResult LoadCustomFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
                          aclrtBinHandle &binHandle)
{
    int32_t deviceLogicId;
    // TODO: 添加返回值校验
    aclrtGetDevice(&deviceLogicId);

    printf("[%s:%d]deviceLogicId is %d\n", __FUNCTION__, __LINE__, deviceLogicId);

    s64 isOpenCustomSwitch = 0;

    printf("[%s:%d]\n", __FUNCTION__, __LINE__);

    CHK_RET(hcalrtGetDeviceInfo(deviceLogicId, ACL_DEV_ATTR_CUST_OP_PRIVILEGE, isOpenCustomSwitch));
    printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    if (isOpenCustomSwitch == 1) {
        printf("[%s:%d]\n", __FUNCTION__, __LINE__);
        HcclResult ret = LoadBinaryFromFile(binPath, optionType, cpuKernelMode, binHandle, true);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[LoadCustomFile]errNo[0x%016llx]load custom file fail, path[%s] optionType[%u]"
                                "cpuKernelMode[%u].",
                                ret, binPath, optionType, cpuKernelMode),
                    ret);
    } else {
        printf("[%s:%d]\n", __FUNCTION__, __LINE__);
        binHandle = nullptr;
        HCCL_RUN_WARNING("[LoadCustomFile]custom switch is not open, please confirm the switch.");
    }
    printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    return HCCL_SUCCESS;
}

// 当前不提供卸载能力，流程上没有点可以卸载
HcclResult LoadAICPUKernel(void)
{
    // 不需要重复加载
    if (g_binKernelHandle != nullptr) {
        return HCCL_SUCCESS;
    }
    // 加载AICPU自定义算子
    const char *binPath1 = "/usr/local/Ascend/latest/scatter_aicpu_kernel.json";
    return LoadCustomFile(binPath1, ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 1, g_binKernelHandle);
}

}