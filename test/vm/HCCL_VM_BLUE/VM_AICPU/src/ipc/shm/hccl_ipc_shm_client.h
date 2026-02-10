#ifndef HCCL_IPC_SHM_CLIENT_H
#define HCCL_IPC_SHM_CLIENT_H

#include <string>
#include <memory>
#include <boost/interprocess/ipc/message_queue.hpp> 
#include "hccl_ipc_client.h"

using namespace boost::interprocess;

class HcclIpcShmClient : public HcclIpcClient {
public:
    HcclIpcShmClient();
    virtual ~HcclIpcShmClient();
    virtual HcclSim::HcclVmResult Connect(std::string& addr) final;
    virtual HcclSim::HcclVmResult Push(const char* msg, const size_t size) final;
private:
    std::shared_ptr<message_queue>  m_mq;
    std::string                     m_addr;
};

#endif