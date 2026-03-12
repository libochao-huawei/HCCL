#ifndef SIM_CHANNEL_MANAGER_H
#define SIM_CHANNEL_MANAGER_H

#include <memory>
#include <unordered_map>
#include "hccl_api.h"
#include "sim_channel.h"
#include "hccl_proxy_pub.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl_sim_shm_manager.h"
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#define CHANNEL_TIME_OUT 10

namespace ipc = boost::interprocess;
namespace bpt = boost::posix_time;

namespace HcclProxy {
const char* const SHM_CHANNEL_CONNECT_BARRIER = "SimChannelConnectBarrier";
class SimChannelMgr {
public:
    static std::string GetChannelKey(std::shared_ptr<SimChannel> channel);

    SimChannelMgr(std::string commId, uint32_t curRank, uint32_t rankSize) : commId_(commId), curRank_(curRank), rankSize_(rankSize) {};
    ~SimChannelMgr() = default;

    HcclResult ChannelCommCreate(const std::string &commId, const char *channelTag, CommEngine engine,
        const HcclChannelDesc *channelDescList, uint32_t listNum, ChannelHandle *channelList);
    
    HcclResult ChannelCommDestroy(ChannelHandle *channelList, uint32_t channelNum);
    HcclResult CommChannelGetHcclBuffer(ChannelHandle channel, CommBuffer *buffer);
    HcclResult MockCommChannelGetHcclBuffer(ChannelHandle channel, CommBuffer *buffer);

private:
    std::string commId_{""};
    uint32_t curRank_{0xFFFFFFFF};
    uint32_t rankSize_{0};

    std::unordered_map<std::string, std::shared_ptr<SimChannel>> channelMap_;
};

struct ChannelConnectBarrier {
    ipc::interprocess_mutex mutex {};
    ipc::interprocess_condition cond;
    uint32_t finCount{0};                     // 当前已到达的进程数

	ChannelConnectBarrier() = default;
	ChannelConnectBarrier(const ChannelConnectBarrier&) = delete;
	ChannelConnectBarrier& operator=(const ChannelConnectBarrier&) = delete;

	void Reset()
	{
		ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
		finCount = 0;
	}

    bool Wait(uint32_t target, uint32_t barrierTimeout) {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
        
        finCount++; 

        if (finCount >= target) {
            cond.notify_all();
            return true;
        }

        bpt::ptime barrierLine = bpt::microsec_clock::universal_time() + bpt::seconds(barrierTimeout);
        while (finCount < target) {
            bool is_notified = cond.timed_wait(lock, barrierLine);

            if (!is_notified) {
                finCount--;
                // 一个进程超时退出，整个任务失败
                cond.notify_all(); 
                return false; // 返回超时状态
            }
        }
        return true;
    }
};

};
#endif  // SIM_CHANNEL_MANAGER_H