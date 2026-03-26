#include <cstdlib>
#include <cerrno>
#include <limits.h>
#include <string>
#include <unistd.h>
#include <mmpa_api.h>
#include "load_kernel.h"

namespace ops_hccl_allgather_2in2out {

thread_local aclrtBinHandle g_binKernelHandle = nullptr;

namespace {

HcclResult GetKernelConfigPath(std::string &configPath)
{
    // 这一层只负责找到 AICPU kernel json 所在目录，不涉及算子语义。
    std::string basePath;
    char *envPath = getenv("ASCEND_HOME_PATH");
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, envPath);
    if (envPath != nullptr) {
        basePath = envPath;
    } else {
        basePath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("[GetKernelConfigPath] ENV ASCEND_HOME_PATH is not set, use default path[%s]", basePath.c_str());
    }

    configPath = basePath + "/opp/vendors/cust/aicpu/config/";
    return HCCL_SUCCESS;
}

HcclResult LoadBinaryFromFile(const char *binPath, aclrtBinHandle &binHandle)
{
    CHK_PRT_RET(binPath == nullptr,
        HCCL_ERROR("[LoadBinaryFromFile] binary path is nullptr"), HCCL_E_PTR);

    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(realpath(binPath, realPath) == nullptr,
        HCCL_ERROR("[LoadBinaryFromFile] invalid path[%s], errno[%d]", binPath, errno), HCCL_E_INTERNAL);

    aclrtBinaryLoadOptions loadOptions = {0};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;

    aclError aclRet = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle);
    CHK_PRT_RET(aclRet != ACL_SUCCESS,
        HCCL_ERROR("[LoadBinaryFromFile] load binary failed, ret[%d], path[%s]", aclRet, realPath),
        HCCL_E_OPEN_FILE_FAILURE);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult LoadAICPUKernel(void)
{
    // Host 侧只需要加载一次 binary handle，后续 launch 直接复用即可。
    if (g_binKernelHandle != nullptr) {
        return HCCL_SUCCESS;
    }

    std::string jsonPath;
    CHK_RET(GetKernelConfigPath(jsonPath));
    jsonPath += "liballgather_2in2out_aicpu_kernel.json";
    return LoadBinaryFromFile(jsonPath.c_str(), g_binKernelHandle);
}

} // namespace ops_hccl_allgather_2in2out
