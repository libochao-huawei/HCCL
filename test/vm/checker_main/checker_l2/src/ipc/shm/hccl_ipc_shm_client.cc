#include "hccl_ipc_shm_client.h"
#include "hccl_common_defs.h"

using namespace HcclSim;

HcclIpcShmClient::HcclIpcShmClient() : m_addr("") {}

HcclIpcShmClient::~HcclIpcShmClient()
{
}

HcclVmResult HcclIpcShmClient::Connect(std::string &addr)
{
    m_addr = addr;
    m_mq = std::make_shared<message_queue>(open_only, m_addr.c_str());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclIpcShmClient::Push(const char *msg, const size_t size)
{
    m_mq->send(msg, size, 0);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}