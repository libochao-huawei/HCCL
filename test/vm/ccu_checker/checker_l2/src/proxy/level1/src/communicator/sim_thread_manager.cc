#include <mutex>
#include "sim_thread_manager.h"

using namespace std;
using namespace hccl;

namespace HcclProxy {

SimThreadMgr::SimThreadMgr(std::string commId, uint32_t curRank) : commId_(commId), curRank_(curRank)
{}

HcclResult SimThreadMgr::CommEngineToNotifyLoadType(CommEngine engine, NotifyLoadType &type)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
            type =  NotifyLoadType::HOST_NOTIFY;
            break;
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
            type =  NotifyLoadType::DEVICE_NOTIFY;
            break;
        default:
            // HCCL_ERROR("[ThreadMgr] Unknown comm engine type: %d", engine);
            printf("[ThreadMgr] Unknown comm engine type: %d \n", engine);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult SimThreadMgr::CommAllocThreadResByStream(
    CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread)
{
    NotifyLoadType notifyLoadType;
    // CHK_RET(CommEngineToNotifyLoadType(engine, notifyLoadType));
    auto ret = CommEngineToNotifyLoadType(engine, notifyLoadType);
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ThreadMgr] [%s] CommEngineToNotifyLoadType fail \n", __func__);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(mainThreadMutex_);
    // std::unique_ptr<SimHcclThread> handle = std::make_unique<SimHcclThread>(stream, notifyNum, notifyLoadType);
    std::unique_ptr<SimHcclThread> handle = std::unique_ptr<SimHcclThread>(new SimHcclThread(stream, notifyNum, notifyLoadType));
    handle->SetCurRank(curRank_);
    handle->SetCtxIndex(0);

    // CHK_RET(handle->Init());
    ret = handle->Init();
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ThreadMgr] [%s] thread init fail \n", __func__);
        return HCCL_E_PARA;
    }
    mainThread_.emplace(stream, std::move(handle));

    *thread = reinterpret_cast<ThreadHandle>(mainThread_[stream].get());
    return HCCL_SUCCESS;
}

HcclResult SimThreadMgr::CommAllocThreadRes(
    HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle *thread)
{
    std::lock_guard<std::mutex> lock(threadMutex_);
    for (uint32_t i = 0; i < threadNum; ++i) {
        auto simThread = std::make_shared<SimHcclThread>(StreamType::STREAM_TYPE_RESERVED, notifyNumPerThread, NotifyLoadType::HOST_NOTIFY);
        simThread->SetCurRank(curRank_);
        simThread->SetCtxIndex(i + 1);
        // CHK_RET(simThread->Init());
        auto ret = simThread->Init();
        if (ret != HcclResult::HCCL_SUCCESS) {
            printf("[ThreadMgr] [%s] thread init fail \n", __func__);
            return HCCL_E_PARA;
        }
        threads_.push_back(simThread);

        thread[i] = reinterpret_cast<ThreadHandle>(simThread.get());
    }
    return HCCL_SUCCESS;
}

HcclResult SimThreadMgr::CommGetNotifyNumInThread(ThreadHandle thread, uint32_t *notifyNum)
{
    SimHcclThread* hcclThread = reinterpret_cast<SimHcclThread*>(thread);
    // CHK_PTR_NULL(hcclThread);
    if (hcclThread == nullptr) {
        return HCCL_E_PTR;
    }
    *notifyNum = hcclThread->GetNotifyNum();
    return HCCL_SUCCESS;
}

};