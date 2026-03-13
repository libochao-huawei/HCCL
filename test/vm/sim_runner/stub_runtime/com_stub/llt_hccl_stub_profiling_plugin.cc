#include "task_profiling_pub.h"
#include "toolchain/prof_api.h"

uint64_t MsprofStr2Id(const char *hashInfo, size_t length)
{
    return 1;
}

int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName)
{
    return 0;
}

uint64_t MsprofSysCycleTime()
{
    return 1;
}

int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle)
{
    return 0;
}

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi *api)
{
    return 0;
}

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    return 0;
}

int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    return 0;
}