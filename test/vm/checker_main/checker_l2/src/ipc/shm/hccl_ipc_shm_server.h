#ifndef HCCL_IPC_SHM_SERVER_H
#define HCCL_IPC_SHM_SERVER_H

#include <string>
#include <memory>
#include "hccl_ipc_server.h"
#include <boost/interprocess/ipc/message_queue.hpp>

using namespace boost::interprocess;

class HcclIpcShmServer : public HcclIpcServer {
public:
    HcclIpcShmServer();
    virtual ~HcclIpcShmServer();
    virtual HcclSim::HcclVmResult Bind(std::string& addr) final;
    virtual HcclSim::HcclVmResult Pull(char* recvBuf, size_t& recvSize) final;
private:
    std::shared_ptr<message_queue>  m_mq;
    std::string                     m_addr;
};

#endif