// 为确保Scatter正常执行打的临时桩，需要适时移除

#include <stdio.h>
#include "hccl/dtype_common.h"
#include <cstring>
#include "hccl_vm_log.h"

namespace ops_hccl {
bool RunIndependentOpExpansion(DevType deviceType)
{
    HCCL_VM_TRACE("[{}] return true", __func__);
    return true;
}

bool IsAiCpuMode(DevType deviceType, u32 rankSize)
{
    const char* env = getenv("HCCL_OP_EXPANSION_MODE");
    HCCL_VM_INFO("[{}] env = {}", __func__, env);
    // 判断是否是AI_CPU
    if (env != nullptr && strcmp(env, "AI_CPU") == 0) {
        return true;
    }
    return false;
}
}

