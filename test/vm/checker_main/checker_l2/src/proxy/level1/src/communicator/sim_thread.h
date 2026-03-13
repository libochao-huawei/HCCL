#ifndef SIM_THREAD_H
#define SIM_THREAD_H

#include "hccl_proxy_pub.h"
#include "hccl_sim_data_defs.h"
#include "hccl_common_defs.h"
#include "hccl_sim_world_pub.h"
#include "acl/acl_base.h"
#include "hccl_common.h"
#include "stream_pub.h"
#include "hccl/hccl_types.h"


namespace HcclProxy {
class SimHcclThread {
public:
    SimHcclThread(aclrtStream rtStream, uint32_t notifyNum, const hccl::NotifyLoadType notifyLoadType);
    SimHcclThread(hccl::StreamType streamType, uint32_t notifyNum, const hccl::NotifyLoadType notifyLoadType);
    ~SimHcclThread();

    HcclResult Init();

    std::string ToString() const;

    void SetCurRank(uint32_t rankId) { curRank_ = rankId; }
    uint32_t GetCurRank() const { return curRank_; };
    uint32_t GetNotifyNum() const { return notifyNum_; }
    ShmSimStream* GetStream() const { return stream_; }
    void SetCtxIndex(uint32_t index) { ctxIndex_ = index; }
    uint64_t GetNotifyIdByIndex(uint32_t notifyIndex) const;

private:
    uint32_t ctxIndex_ = 0;
    uint32_t curRank_ = 0;
    uint32_t notifyNum_ = 0;
    ShmSimStream* stream_ = nullptr;
    hccl::StreamType streamType_ = hccl::StreamType::STREAM_TYPE_RESERVED;
    // std::vector<ShmSimNotify*> notifys_;
    std::vector<uint64_t> notifys_; // notifyId
    hccl::NotifyLoadType notifyLoadType_ = hccl::NotifyLoadType::HOST_NOTIFY;
};  // class SimHcclThread

};
#endif  // SIM_COMMUNICATOR_H