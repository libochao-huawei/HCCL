#ifndef OPS_HCCL_CUSTOM_ALLGATHER_BATCH_CCU_RESOURCE_FLOW_H
#define OPS_HCCL_CUSTOM_ALLGATHER_BATCH_CCU_RESOURCE_FLOW_H

#include <memory>
#include <vector>

#include "ccu_kernel.h"
#include "common.h"

namespace ops_hccl_allgather_batch {

struct CcuKernelInfo {
    uint32_t resGroup = 0;
    hcomm::KernelCreator creator;
    std::shared_ptr<hcomm::CcuKernelArg> kernelArg;
    std::vector<HcclChannelDesc> channels;
};

struct AlgResourceRequest {
    uint32_t notifyNumOnMainThread = 0;
    uint32_t slaveThreadNum = 0;
    std::vector<uint32_t> notifyNumPerThread;
    std::vector<CcuKernelInfo> ccuKernelInfos;
    std::vector<uint32_t> ccuKernelNum;
};

struct AlgResourceCtx {
    std::vector<ThreadHandle> threads;
    std::vector<uint32_t> ccuKernelNum;
    std::vector<CcuKernelHandle> ccuKernels;
};

HcclResult HcclGetThreadForCcu(HcclComm comm, aclrtStream stream, AlgResourceRequest &resRequest, AlgResourceCtx &resCtx);
HcclResult HcclGetChannelForCcu(HcclComm comm, AlgResourceRequest &resRequest);
HcclResult HcclGetCcuKernel(HcclComm comm, AlgResourceRequest &resRequest, AlgResourceCtx &resCtx);
HcclResult HcclAllocAlgResourceCcu(HcclComm comm, aclrtStream stream, AlgResourceRequest &resRequest, AlgResourceCtx &resCtx);

} // namespace ops_hccl_allgather_batch

#endif
