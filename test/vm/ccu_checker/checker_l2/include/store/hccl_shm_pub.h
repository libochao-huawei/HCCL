#ifndef HCCLSIM_SHM_PUB_H
#define HCCLSIM_SHM_PUB_H

#include "hccl_common_defs.h"
#include "hccl_sim_world_pub.h"
#include "hccl_sim_data_defs.h"

HcclSim::HcclVmResult ShmEnvInit();

// Virtual Runtime
HcclSim::HcclVmResult GetAddrByOffset(uint64_t offset, void** addr);

HcclSim::HcclVmResult GetTaskCollectionByCid(uint64_t taskCid, HcclTaskMetaData* task);

// Proxy
HcclSim::HcclVmResult AllocRankMemForProxy(uint64_t size, void** memAddr);

HcclSim::HcclVmResult GetProxyBufferMemOffset(const void* addr, uint64_t* offset);
HcclSim::HcclVmResult GetMockBufferMemOffset(const void* addr, uint64_t* offset);

HcclSim::HcclVmResult GetRankOffset(uint32_t rankId, const void* addr, uint64_t* offset);   // todo 可能不需要
HcclSim::HcclVmResult InsertProxyBaseAddr(uint32_t rankId);  // todo 可能不需要

HcclSim::HcclVmResult InsertTaskToCollection(HcclTaskMetaData* task, uint32_t* index);

// Host
HcclSim::HcclVmResult InitIpc(uint32_t rankNum, uint32_t serverNum = 0);

HcclSim::HcclVmResult InitSharedMemory(TopoMeta topoMeta);

// Plugin
HcclSim::HcclVmResult GetTaskCollection(HcclTaskMetaData* task, uint32_t* len);
HcclSim::HcclVmResult GetMemLayout(std::vector<DumpMemBlock> &memLayout);

HcclSim::HcclVmResult AllocatePhy(void **ptr, uint64_t *offset_ptr, size_t size);
HcclSim::HcclVmResult DeallocatePhy(void *ptr, uint64_t offset_ptr, size_t size);
HcclSim::HcclVmResult GetPhyPtrFromOffsetPtr(void **ptr, uint64_t offset_ptr);

HcclSim::HcclVmResult AllocateVir(void **ptr, size_t size);
HcclSim::HcclVmResult DeallocateVir(void *ptr, size_t size);

#endif