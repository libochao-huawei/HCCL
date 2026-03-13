#ifndef SIM_THREAD_MANAGER_H
#define SIM_THREAD_MANAGER_H

// #include "topo_model.h"
// #include "comm.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_data_defs.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl_common_defs.h"
#include "acl/acl_base.h"
#include "sim_thread.h"


namespace HcclProxy {

class SimThreadMgr {
public:
    SimThreadMgr(std::string commId, uint32_t curRank);
    ~SimThreadMgr() = default;
    HcclResult CommAllocThreadRes(HcclComm comm, CommEngine engine, uint32_t threadNum,
        uint32_t notifyNumPerThread, ThreadHandle *thread);

    HcclResult CommAllocThreadResByStream(CommEngine engine,
        aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread);

    HcclResult CommGetNotifyNumInThread(ThreadHandle thread, uint32_t *notifyNum);

private:
    HcclResult CommEngineToNotifyLoadType(CommEngine engine, hccl::NotifyLoadType &type);

    std::string commId_;
    uint32_t curRank_;

    std::mutex threadMutex_;
    std::vector<std::shared_ptr<SimHcclThread>> threads_;

    std::mutex mainThreadMutex_;
    std::map<rtStream_t, std::unique_ptr<SimHcclThread>> mainThread_;
};  // class SimThreadMgr

};
#endif  // SIM_COMMUNICATOR_H