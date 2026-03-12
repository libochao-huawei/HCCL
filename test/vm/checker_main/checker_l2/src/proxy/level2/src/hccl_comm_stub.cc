#include <unistd.h>
#include <vector>
#include <atomic>
#include <stdio.h>
#include <pthread.h>
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "runtime/base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "hccl_sim_shm_manager.h"
#include "task_status_cache.h"
// #include "hccl_vm.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "sim_runner_ops.h"
#include "hccp_common.h"
#include "ip_address.h"
#include "sim_runner_common.h"
#include "hccl_vm_log.h"


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

extern HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);
extern HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);
extern HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm);
extern HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm);
extern HcclResult HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank,
    HcclCommConfig *config, HcclComm *comm);

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo)
{
    HCCL_VM_DEBUG("HcclGetRootInfo stub ...");
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm)
{
    HCCL_VM_DEBUG("HcclCommInitRootInfo stub  ...");
    
    return HcclCommInitClusterInfo("./ranktable.json", nRanks, comm);
}

HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm)
{
    HCCL_VM_DEBUG("HcclCommInitRootInfoConfig stub ...");
    HcclCommConfig *cfg = const_cast<HcclCommConfig *>(config);
    return HcclCommInitClusterInfoConfig("./ranktable.json", nRanks, cfg, comm);
}

#ifdef __cplusplus
}
#endif  // __cplusplus