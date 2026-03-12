
#ifndef ADAPTER_V2_RDMA_HANDLE_MANAGER_STUB_H
#define ADAPTER_V2_RDMA_HANDLE_MANAGER_STUB_H

#include "orion_adapter_hccp.h"
#include <map>

namespace Hccl {

extern std::map<RdmaHandle, std::pair<uint32_t, uint32_t>> g_rdmaHandle2DieIdAndFuncId;
extern RdmaHandle g_rdmaHandlePtr;

}

#endif

