#include <chrono>
#include <thread>
#include "sim_channel_manager.h"

using namespace std;
using namespace HcclSim;

namespace HcclProxy {

std::string SimChannelMgr::GetChannelKey(std::shared_ptr<SimChannel> channel)
{
    return channel->GetTag() + ":" + to_string(channel->GetEngine()) + ":" +
        to_string(channel->GetRmtRankId()) + ":" + to_string(channel->GetProtocol());
}

HcclResult SimChannelMgr::ChannelCommCreate(const std::string &commId, const char *channelTag, CommEngine engine,
    const HcclChannelDesc *channelDescList, uint32_t listNum, ChannelHandle *channelList)
{
    vector<pair<uint32_t, shared_ptr<SimChannel>>> tmpChannels{};

    for (uint32_t i = 0; i < listNum; ++i) {
        string channelKey = string(channelTag) + ":" + to_string(engine) + ":" +
            to_string(channelDescList[i].remoteRank) + ":" + to_string(channelDescList[i].channelProtocol);
        if (channelMap_.find(channelKey) != channelMap_.end()) {
            // 已存在缓存
            channelList[i] = reinterpret_cast<ChannelHandle>(channelMap_[channelKey].get());
        } else {
            // 不存在缓存，创建新的channel
            shared_ptr<SimChannel> channel = make_shared<SimChannel>(commId, string(channelTag), engine, channelDescList[i].channelProtocol,
                curRank_, channelDescList[i].remoteRank, channelDescList[i].notifyNum);
            // CHK_RET(channel->Init());
            auto ret = channel->Init();
            if (ret != HcclResult::HCCL_SUCCESS) {
                printf("[ERROR] [SimChannelMgr::%s] channel init fail \n", __func__);
                return HcclResult::HCCL_E_PARA;
            }
            tmpChannels.push_back({i, channel});
        }
    }

    // 模拟建链
    try {
        // 发送channel本rank资源
        for (auto& pair : tmpChannels) {
            auto tmpChannel = pair.second;
            tmpChannel->loc2rmt_MQ = "MQ_" + to_string(tmpChannel->GetLocRankId()) + "_TO_" + to_string(tmpChannel->GetRmtRankId());
            const char* MQ_LOC_TO_RMT = (tmpChannel->loc2rmt_MQ).c_str();
            ipc::message_queue::remove(MQ_LOC_TO_RMT);
            ipc::message_queue mqSend(ipc::create_only, MQ_LOC_TO_RMT, 1, sizeof(ResExchangeMessage));
            ResExchangeMessage localRes;
            localRes.memBlockIdx = tmpChannel->GetLocMem();
            localRes.notifyNum = tmpChannel->GetNotifyNum();
            auto locNotifys = tmpChannel->GetLocNotifys();
            for (int i = 0; i < localRes.notifyNum; ++i) {
                localRes.notifyIds[i] = locNotifys[i];
            }
            std::cout << tmpChannel->GetLocRankId() <<" Sending data to " << tmpChannel->GetRmtRankId() << "..." << std::endl;
            mqSend.send(&localRes, sizeof(localRes), 0);
        }
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] : ChannelCommCreate : send data exception: " << e.what() << std::endl;
        for (auto& pair : tmpChannels) {
            const char* MQ_LOC_TO_RMT = pair.second->loc2rmt_MQ.c_str();
            ipc::message_queue::remove(MQ_LOC_TO_RMT);
        }
        return HcclResult::HCCL_E_PARA;
    }

    // SHM内创建共享barrier
    ChannelConnectBarrier* barrierPtr = SHMManager::GetSegment().find_or_construct<ChannelConnectBarrier>(SHM_CHANNEL_CONNECT_BARRIER)();
    if(barrierPtr->Wait(rankSize_, CHANNEL_TIME_OUT)) {
        std::cout << "Rank : " << curRank_ << " send MessageQueue Barrier passed" << std::endl;
    } else {
        std::cout << "Rank : " << curRank_ << " send MessageQueue Barrier timeout! Aborting..." << std::endl;
        return HcclResult::HCCL_E_TIMEOUT;
    }

    ResExchangeMessage rmtRes;
    ipc::message_queue::size_type recSize;
    unsigned int prio;
    try {
        for (auto& pair : tmpChannels) {
            auto tmpChannel = pair.second;
            tmpChannel->rmt2loc_MQ = "MQ_" + to_string(tmpChannel->GetRmtRankId()) + "_TO_" + to_string(tmpChannel->GetLocRankId());
            const char* MQ_RMT_TO_LOC = (tmpChannel->rmt2loc_MQ).c_str();
            ipc::message_queue mqRecv(ipc::open_only, MQ_RMT_TO_LOC);
            // 尝试交换资源
            bpt::ptime deadline = bpt::microsec_clock::universal_time() + bpt::seconds(CHANNEL_TIME_OUT);
            std::cout << tmpChannel->GetLocRankId() <<" waitting data from " << tmpChannel->GetRmtRankId();
            std::cout << " for most : " << CHANNEL_TIME_OUT << " seconds"<< std::endl;
            bool received = mqRecv.timed_receive(&rmtRes, sizeof(rmtRes), recSize, prio, deadline);
            if (!received) {
                printf("[ERROR] [SimChannelMgr::%s] wait channel remote resourse timeout", __func__);
                ipc::message_queue::remove(MQ_RMT_TO_LOC);
                return HcclResult::HCCL_E_TIMEOUT;
            }
            std::cout << "[MQ_SUCCESS] " << tmpChannel->GetLocRankId() <<" get data from " << tmpChannel->GetRmtRankId() << std::endl;
            ipc::message_queue::remove(MQ_RMT_TO_LOC);
            tmpChannel->SetRmtMem(rmtRes.memBlockIdx);
            tmpChannel->SetRmtNotifys(rmtRes.notifyIds, rmtRes.notifyNum);
        }
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] : ChannelCommCreate : reveive data exception: " << e.what() << std::endl;
        for (auto& pair : tmpChannels) {
            const char* MQ_RMT_TO_LOC = pair.second->rmt2loc_MQ.c_str();
            ipc::message_queue::remove(MQ_RMT_TO_LOC);
        }
        return HcclResult::HCCL_E_PARA;
    }

    for (auto& pair : tmpChannels) {
        string channelKey = SimChannelMgr::GetChannelKey(pair.second);
        channelMap_.insert({channelKey, pair.second});
        channelList[pair.first] = reinterpret_cast<ChannelHandle>(channelMap_[channelKey].get());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimChannelMgr::ChannelCommDestroy(ChannelHandle *channelList, uint32_t channelNum) {
    for (int i = 0; i < channelNum; ++i) {
        SimChannel* channelPtr = reinterpret_cast<SimChannel*>(channelList[i]);
        string channelKey = channelPtr->GetTag() + ":" + to_string(channelPtr->GetEngine()) + ":" +
                            to_string(channelPtr->GetRmtRankId()) + ":" + to_string(channelPtr->GetProtocol());
        auto ret = channelPtr->Destrory();
        if (ret != HcclResult::HCCL_SUCCESS) {
            printf("[ERROR] [SimChannelMgr::%s] channel destrory fail \n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
        channelMap_.erase(channelKey);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimChannelMgr::CommChannelGetHcclBuffer(ChannelHandle channel, CommBuffer *buffer) {
    SimChannel* channelPtr = reinterpret_cast<SimChannel*>(channel);
    if (channelPtr == nullptr) {
        printf("[ERROR] [SimChannelMgr::%s] ChannelHandle is NULL \n", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t rmtRankId = channelPtr->GetRmtRankId();
    size_t rmtMemIdx = channelPtr->GetRmtMem();

    ShmSimNpu* rmtNpu = nullptr;
    auto ret = GetNpuByRankId(rmtRankId, &rmtNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimChannelMgr::%s] get npu fail\n", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    buffer->addr = (rmtNpu->memory[rmtMemIdx].addr).get();
    buffer->size = rmtNpu->memory[rmtMemIdx].size;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimChannelMgr::MockCommChannelGetHcclBuffer(ChannelHandle channel, CommBuffer *buffer) {
    SimChannel* channelPtr = reinterpret_cast<SimChannel*>(channel);
    if (channelPtr == nullptr) {
        printf("[ERROR] [SimChannelMgr::%s] ChannelHandle is NULL \n", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t rmtRankId = channelPtr->GetRmtRankId();
    size_t rmtMemIdx = channelPtr->GetRmtMem();

    ShmSimNpu* rmtNpu = nullptr;
    auto ret = GetNpuByRankId(rmtRankId, &rmtNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimChannelMgr::%s] get npu fail\n", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    buffer->addr = reinterpret_cast<void*>(rmtNpu->memory[rmtMemIdx].mockAddr);
    buffer->size = rmtNpu->memory[rmtMemIdx].size;
    return HcclResult::HCCL_SUCCESS;
}
}