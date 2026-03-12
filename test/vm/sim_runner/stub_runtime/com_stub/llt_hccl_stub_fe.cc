#include "platform/platform_infos_def.h"

namespace fe {
uint32_t PlatFormInfos::GetCoreNum() const
{
    return 2;
}

void PlatFormInfos::SetCoreNum(const uint32_t &core_num)
{
    core_num_ = core_num;
}
}  // namespace fe
