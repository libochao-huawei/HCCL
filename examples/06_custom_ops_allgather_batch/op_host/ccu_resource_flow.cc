#include "ccu_resource_flow.h"

#include <hccl_ccu_res.h>

namespace ops_hccl_allgather_batch {

HcclResult HcclGetThreadForCcu(HcclComm comm, aclrtStream stream, AlgResourceRequest &resRequest, AlgResourceCtx &resCtx)
{
    ThreadHandle thread = 0;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream,
                                        resRequest.notifyNumOnMainThread, &thread));
    resCtx.threads.push_back(thread);

    uint32_t maxNotifyNum = 0;
    for (uint32_t notifyNum : resRequest.notifyNumPerThread) {
        if (notifyNum > maxNotifyNum) {
            maxNotifyNum = notifyNum;
        }
    }

    if (resRequest.slaveThreadNum > 0) {
        std::vector<ThreadHandle> threads(resRequest.slaveThreadNum);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, resRequest.slaveThreadNum,
                                  maxNotifyNum, threads.data()));
        resCtx.threads.insert(resCtx.threads.end(), threads.begin(), threads.end());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelForCcu(HcclComm comm, AlgResourceRequest &resRequest)
{
    for (CcuKernelInfo &kernelInfo : resRequest.ccuKernelInfos) {
        std::vector<HcclChannelDesc> &kernelChannelRequest = kernelInfo.channels;
        uint32_t channelNum = kernelChannelRequest.size();
        std::vector<ChannelHandle> kernelChannels(channelNum);

        if (channelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, kernelChannelRequest.data(),
                                       channelNum, kernelChannels.data()));
        }
        kernelInfo.kernelArg->channels = kernelChannels;
        HCCL_INFO("[HcclGetChannelForCcu] Get [%u] channels", channelNum);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCcuKernel(HcclComm comm, AlgResourceRequest &resRequest, AlgResourceCtx &resCtx)
{
    uint32_t totalKernelNum = 0;
    for (uint32_t kernelNum : resRequest.ccuKernelNum) {
        totalKernelNum += kernelNum;
    }
    CHK_PRT_RET(totalKernelNum != resRequest.ccuKernelInfos.size(),
                HCCL_ERROR("[HcclGetCcuKernel] ccuKernel num not match"),
                HCCL_E_INTERNAL);

    uint32_t currentResGroup = 0;
    uint32_t maxResGroup = 0;
    resCtx.ccuKernels.resize(totalKernelNum);
    while (currentResGroup <= maxResGroup) {
        for (uint32_t i = 0; i < totalKernelNum; ++i) {
            CcuKernelInfo &kernelInfo = resRequest.ccuKernelInfos[i];
            if (kernelInfo.resGroup > maxResGroup) {
                maxResGroup = kernelInfo.resGroup;
            }
            if (kernelInfo.resGroup != currentResGroup) {
                continue;
            }

            void *kernelArgPtr = static_cast<void *>(kernelInfo.kernelArg.get());
            void *creatorPtr = static_cast<void *>(&kernelInfo.creator);
            HCCL_INFO("[HcclGetCcuKernel] registering kernel[%u], creatorPtr=%p kernelArgPtr=%p",
                      i, creatorPtr, kernelArgPtr);

            CcuKernelHandle handle = 0;
            CHK_RET(HcclCcuKernelRegister(comm, &handle, creatorPtr, kernelArgPtr));
            resCtx.ccuKernels[i] = handle;
        }
        CHK_RET(HcclCcuKernelRegisterFinish(comm));
        ++currentResGroup;
    }
    resCtx.ccuKernelNum = resRequest.ccuKernelNum;
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceCcu(HcclComm comm, aclrtStream stream, AlgResourceRequest &resRequest, AlgResourceCtx &resCtx)
{
    CHK_RET(HcclGetThreadForCcu(comm, stream, resRequest, resCtx));
    CHK_RET(HcclGetChannelForCcu(comm, resRequest));
    CHK_RET(HcclGetCcuKernel(comm, resRequest, resCtx));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_allgather_batch
