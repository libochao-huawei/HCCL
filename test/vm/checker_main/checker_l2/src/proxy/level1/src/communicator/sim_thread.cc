#include <sstream>
#include "sim_thread.h"

// using namespace hccl;
using namespace std;
using namespace HcclSim;

namespace HcclProxy {

SimHcclThread::SimHcclThread(aclrtStream rtStream, uint32_t notifyNum, const hccl::NotifyLoadType notifyLoadType)
    : notifyNum_(notifyNum), notifyLoadType_(notifyLoadType)
{
    stream_ = reinterpret_cast<ShmSimStream*>(rtStream);
}

SimHcclThread::SimHcclThread(hccl::StreamType streamType, uint32_t notifyNum, const hccl::NotifyLoadType notifyLoadType)
    : streamType_(streamType), notifyNum_(notifyNum), notifyLoadType_(notifyLoadType)
{
}

SimHcclThread::~SimHcclThread()
{
    if (stream_ != nullptr) {
        auto ret = ReleaseStream(stream_->streamId.value);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[SimHcclThread] 析构失败 : release stream fail!");
            stream_->streamId.ToString();
        }
    }

    for (auto& notify : notifys_) {
        auto ret = ReleaseNotify(notify);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[SimHcclThread] 析构失败 : release notify fail!");
            ShmNpuResId notifyResId(notify);
            notifyResId.ToString();
        }
    }
    notifys_.clear();
}

HcclResult SimHcclThread::Init()
{
    if (stream_ == nullptr) {
        void* streamPtr = nullptr;
        auto ret = AllocStream(curRank_, &streamPtr);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [SimHcclThread::%s] AllocStream fail \n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
        stream_ = reinterpret_cast<ShmSimStream*>(streamPtr);
    }

    for (uint32_t i = 0; i < notifyNum_; ++i) {
        void* notifyPtr = nullptr;
        auto ret = AllocNotify(curRank_, &notifyPtr);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [SimHcclThread::%s] AllocNotify fail \n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
        uint64_t notifyId = reinterpret_cast<ShmSimNotify*>(notifyPtr)->notifyId.value;
        notifys_.push_back(notifyId);
    }

    return HcclResult::HCCL_SUCCESS;
}

std::string SimHcclThread::ToString() const
{
    stringstream ss;
    ss << "Thread{";
    ss << "Rank[" << to_string(curRank_) << "]" << ", ";
    ss << "Idx[" << to_string(ctxIndex_) << "]";
    ss << "}";
    return ss.str();
}

uint64_t SimHcclThread::GetNotifyIdByIndex(uint32_t notifyIndex) const
{
    if (notifyIndex >= notifys_.size()) {
        THROW<InvalidParamsException>("[SimHcclThread::%s] notifyIndex[%u] out of bound", __func__, notifyIndex);
    }
    return notifys_[notifyIndex];
}

};