#ifndef HCCLV2_COMMON_INTERFACE_H
#define HCCLV2_COMMON_INTERFACE_H

#include "task_param.h"
#include <vector>
#include <string>

namespace Hccl {
    void DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo> &profInfos);
    void CheckProfilingInfo(const std::vector<CcuProfilingInfo> &profInfos, uint32_t sqeCnt, uint32_t ckeCnt, uint32_t loopGroupCnt);

}  // namespace Hccl

#endif // HCCLV2_COMMON_INTERFACE_H