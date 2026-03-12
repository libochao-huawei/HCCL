#include <map>
#include "com_stub.h"
#include "hccl_shm_pub.h"

using namespace HcclSim;

// Virtual Runtime
HcclVmResult GetAddrByOffset(uint64_t offset, void** addr)
{
    *addr = reinterpret_cast<void*>(offset + GetBaseAddr(offset));
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetTaskCollectionByCid(uint64_t taskCid, HcclTaskMetaData* task)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// Proxy
HcclVmResult AllocRankMemForProxy(uint64_t size, void** memAddr)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetRankOffset(uint32_t rankId, void* addr, uint64_t* offset)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult InsertTaskToCollection(HcclTaskMetaData* task, uint32_t* index)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult InsertProxyBaseAddr(uint32_t rankId)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// Host
HcclVmResult InitIpc(uint32_t rankNum, uint32_t serverNum)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult InitSharedMemory(TopoMeta topoMeta)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// Plugin
HcclVmResult GetTaskCollection(HcclTaskMetaData* task)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}