#include <mmpa_api.h>
#include <vector>`r`n#include <cstdlib>
#include <string>
#include "load_kernel.h"

namespace ops_hccl_double_allgather {

thread_local aclrtBinHandle g_binKernelHandle = nullptr;

static HcclResult TryLoadBinary(const std::string &path)
{
    aclrtBinaryLoadOptions loadOptions = {0};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    aclError aclRet = aclrtBinaryLoadFromFile(path.c_str(), &loadOptions, &g_binKernelHandle);
    if (aclRet == ACL_SUCCESS) {
        HCCL_INFO("load aicpu json success, path=%s", path.c_str());
        return HCCL_SUCCESS;
    }
    return HCCL_E_OPEN_FILE_FAILURE;
}

HcclResult LoadAICPUKernel(void)
{
    if (g_binKernelHandle != nullptr) {
        return HCCL_SUCCESS;
    }

    std::vector<std::string> candidates;
    char *ascendHome = getenv("ASCEND_HOME_PATH");
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, ascendHome);
    if (ascendHome != nullptr) {
        candidates.emplace_back(std::string(ascendHome) + "/opp/vendors/cust/aicpu/config/libdouble_allgather_aicpu_kernel.json");
    }
    candidates.emplace_back("libdouble_allgather_aicpu_kernel.json");
    candidates.emplace_back("../op_kernel_aicpu/libdouble_allgather_aicpu_kernel.json");
    candidates.emplace_back("../build/op_kernel_aicpu/libdouble_allgather_aicpu_kernel.json");
    candidates.emplace_back("../build/libdouble_allgather_aicpu_kernel.json");
    candidates.emplace_back("../op_kernel_aicpu/libdouble_allgather_aicpu_kernel.json");

    for (const auto &candidate : candidates) {
        if (TryLoadBinary(candidate) == HCCL_SUCCESS) {
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("failed to load aicpu kernel json from all candidates");
    return HCCL_E_OPEN_FILE_FAILURE;
}

}

