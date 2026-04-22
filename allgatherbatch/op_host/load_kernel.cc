#include "load_kernel.h"

#include <cstdlib>
#include <limits.h>
#include <string>
#include <unistd.h>

#include "mmpa_api.h"

namespace ops_hccl_allgatherbatch {

namespace {

const char *GetCustomOpsVendor()
{
#ifdef CUSTOM_OPS_VENDOR_STR
    if (CUSTOM_OPS_VENDOR_STR[0] != '\0') {
        return CUSTOM_OPS_VENDOR_STR;
    }
#endif
    return "cust";
}

}  // namespace

thread_local aclrtBinHandle g_allGatherBatchKernelHandle = nullptr;

HcclResult GetKernelConfigPath(std::string &jsonPath)
{
    char *ascendHome = getenv("ASCEND_HOME_PATH");
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, ascendHome);

    std::string basePath;
    if (ascendHome != nullptr) {
        basePath = ascendHome;
    } else {
        basePath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("ASCEND_HOME_PATH is not set, fallback to %s", basePath.c_str());
    }

    const char *vendor = GetCustomOpsVendor();
    jsonPath = basePath + "/opp/vendors/" + vendor + "/aicpu/config/liballgatherbatch_aicpu_kernel.json";
    return HCCL_SUCCESS;
}

static HcclResult LoadBinaryFromFile(const char *jsonPath, aclrtBinHandle &binHandle)
{
    HCCL_CHK_PTR(jsonPath);

    char realPath[PATH_MAX] = {0};
    if (realpath(jsonPath, realPath) == nullptr) {
        HCCL_ERROR("kernel json path is invalid: %s", jsonPath);
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    aclrtBinaryLoadOptions loadOptions = {0};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;

    aclError aclRet = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("aclrtBinaryLoadFromFile failed, ret=%d", static_cast<int>(aclRet));
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    return HCCL_SUCCESS;
}

HcclResult LoadAICPUKernel()
{
    if (g_allGatherBatchKernelHandle != nullptr) {
        return HCCL_SUCCESS;
    }

    std::string jsonPath;
    HCCL_CHK_RET(GetKernelConfigPath(jsonPath));
    HCCL_CHK_RET(LoadBinaryFromFile(jsonPath.c_str(), g_allGatherBatchKernelHandle));
    HCCL_INFO("loaded allgatherbatch aicpu kernel json: %s", jsonPath.c_str());
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch