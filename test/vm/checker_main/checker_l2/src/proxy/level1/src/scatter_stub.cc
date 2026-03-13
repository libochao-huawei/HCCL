// 为确保Scatter正常执行打的临时桩，需要适时移除

#include <stdio.h>
#include "hccl/dtype_common.h"
#include <cstring>

namespace ops_hccl {
bool RunIndependentOpExpansion(DevType deviceType)
{
    printf("[ScatterStub][RunIndependentOpExpansion] return true.\n");
    return true;
}

bool IsAiCpuMode(DevType deviceType, u32 rankSize)
{
    const char* env = getenv("HCCL_OP_EXPANSION_MODE");
    printf("[ScatterStub][IsAiCpuMode] env = %s.\n", env);
    // 判断是否是AI_CPU
    if (env != nullptr && strcmp(env, "AI_CPU") == 0) {
        return true;
    }
    return false;
}
}

