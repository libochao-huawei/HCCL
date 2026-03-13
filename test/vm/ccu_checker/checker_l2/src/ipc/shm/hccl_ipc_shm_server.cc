#include "hccl_ipc_shm_server.h"
#include "hccl_common_defs.h"

using namespace HcclSim;

HcclIpcShmServer::HcclIpcShmServer() : m_addr("") {}

HcclIpcShmServer::~HcclIpcShmServer()
{
}

HcclVmResult HcclIpcShmServer::Bind(std::string &addr)
{
    m_addr = addr;
    m_mq = std::make_shared<message_queue>(open_only, m_addr.c_str());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclIpcShmServer::Pull(char *recvBuf, size_t &recvSize)
{
    unsigned int priority = 0;
    m_mq->receive(recvBuf,  recvSize, recvSize, priority);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}