#ifndef SIM_CHANNEL_H
#define SIM_CHANNEL_H

#include <memory>
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl_sim_data_defs.h"
#include "hccl_sim_world_pub.h"
#include "hccl_proxy_pub.h"

// #include "sim_task.h"

namespace HcclProxy {

struct ResExchangeMessage {
    size_t memBlockIdx;
    size_t notifyNum;
    uint64_t notifyIds[MAX_NOTIFY_NUM];
};

class SimChannel {
public:
    SimChannel(const std::string& commId, const std::string& tag, CommEngine engine, CommProtocol protocol,
        uint32_t locRankId, uint32_t rmtRankId, uint32_t notifyNum);
    ~SimChannel();

    HcclResult Init();
    HcclResult Destrory();

    std::string ToString() const;

    const std::string& GetCommId() const { return commId_; };
    const std::string& GetTag() const { return tag_; };
    CommEngine GetEngine() const { return engine_; };
    CommProtocol GetProtocol() const { return protocol_; };
    // LinkProtoStub GetLinkType() const;
    uint32_t GetLocRankId() const { return locRankId_; };
    uint32_t GetRmtRankId() const { return rmtRankId_; };
    size_t GetLocMem() const { return locMemIdx_; };
    size_t GetRmtMem() const { return rmtMemIdx_; };
    size_t GetNotifyNum() const { return notifyNum_; };
    const std::vector<uint64_t>& GetLocNotifys() const { return locNotifys_; };
    uint64_t GetLocNotifyIdByIndex(uint32_t notifyIdx) const;
    uint64_t GetRmtNotifyIdByIndex(uint32_t notifyIdx) const;
    void SetRmtMem(size_t rmtMemIdx) { rmtMemIdx_ = rmtMemIdx; };
    void SetRmtNotifys(uint64_t* notifyIds, size_t nums);

    std::string loc2rmt_MQ{""};
    std::string rmt2loc_MQ{""};

private:
    std::string commId_{""};
    std::string tag_{""};
    CommEngine engine_{CommEngine::COMM_ENGINE_RESERVED};
    CommProtocol protocol_{CommProtocol::COMM_PROTOCOL_RESERVED};

    uint32_t locRankId_{0xFFFFFFFF};
    uint32_t rmtRankId_{0xFFFFFFFF};
    size_t locMemIdx_{MAX_MEM_BLOCK_NUM};   // local CCL Buffer
    size_t rmtMemIdx_{MAX_MEM_BLOCK_NUM};   // remote CCL Buffer
    uint32_t memSize {400*1024*1024}; // MB
    uint32_t notifyNum_{0};
    std::vector<uint64_t> locNotifys_; // notifyId
    std::vector<uint64_t> rmtNotifys_; // notifyId

    // bool isReady_{false};
};

}   // namespace HcclSim
#endif // SIM_MEM_LAYOUT_H