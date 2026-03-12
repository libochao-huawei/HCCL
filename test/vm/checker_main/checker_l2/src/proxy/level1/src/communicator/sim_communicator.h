#ifndef SIM_COMMUNICATOR_H
#define SIM_COMMUNICATOR_H

#include "topo_model.h"
#include "rank_table.h"
#include "hccl_proxy_pub.h"
#include "sim_context_manager.h"
#include "sim_thread_manager.h"
#include "sim_channel.h"
#include "sim_channel_manager.h"
#include "hccl_sim_data_defs.h"

namespace HcclProxy {

HcclResult Sim_HcclCommInitClusterInfo(ShmCommDomain *commDomain, uint32_t rank, HcclComm *comm);

class SimCommunicator {
public:
    explicit SimCommunicator() = default;
    ~SimCommunicator() = default;

public:
    HcclResult Init(const char *clusterInfo, uint32_t rank);
    HcclResult Init(ShmCommDomain *commDomain, uint32_t rank, uint32_t rankSize);
    HcclResult MockInit(ShmCommDomain *commDomain, uint32_t rank, uint32_t rankSize);
    uint32_t GetRankId();
    uint32_t GetRankSize();
    std::string GetIdentifier();
    HcclResult GetCommRankGraph(void **graph, uint32_t *len);
    HcclResult GetHcclBuffer(CommBuffer *buffer);
    // HcclResult ChannelCommCreate(const std::string &commId, const std::string &tag, CommEngine engine, 
    //     const ChannelDesc *channelDescList, uint32_t listNum, ChannelHandle *channelList);
    // HcclResult ChannelCommGetHcclBuffer(ChannelHandle channel, CommBuffer *buffer);

private:
    HcclResult GetFileRealPath(const char *rankTable, std::string &realFilePath);
    HcclResult ParseRankTable(const char *clusterInfo);

    HcclResult GetDefaultCommConfig(HcclCommConfig &commConfig, const std::string &commName) const;
    HcclResult SetIndependentOpConfig(const HcclCommConfig &commConfig);

public:
    std::unique_ptr<TopoModel> topoModel_;
    std::unique_ptr<SimContextMgr> contextManager_{nullptr};
    std::unique_ptr<SimThreadMgr> independentOpThreadMgr_{nullptr};
    std::unique_ptr<SimChannelMgr> channelMgr_{nullptr};

private:
    // config内容
    int32_t commEngine_ = -1;
    uint32_t threadNum_ = 0;
    uint32_t notifyNumPerThread_ = 0;
    void* cclBufferAddr_;
    uint64_t cclBufferSize_{400*1024*1024};
    std::string commId_;

    uint32_t curRank_{0};
    uint32_t rankSize_{0};
    std::string identifier_{""};
}; // SimCommunicator

};

#endif  // SIM_COMMUNICATOR_H