#include "hccl_ipc_comm.h"
#include "hccl_ipc_defs.h"
#include "hccl_ipc_factory.h"
#include "hccl_common_defs.h"
#include "hccl_dependency.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace HcclSim;

HcclIpcComm::HcclIpcComm() : m_mode(0)
{
    std::ifstream procComm("/proc/self/comm");
    if (!procComm.is_open()) {
        throw std::runtime_error("Failed to open /proc/self/comm");
    }
    std::string processName;
    procComm >> processName;

    if (processName.find("virtual") != std::string::npos) {
        m_mode = 1;
    }
}

HcclIpcComm::~HcclIpcComm()
{
    if (m_puller != nullptr) {
        delete m_puller;
    }

    for (auto& pusher : m_pushers) {
        delete pusher.second;
    }
    m_pushers.clear();
}

HcclVmResult HcclIpcComm::Push(const HcclTaskReq& req)
{
    if (m_mode) {
        return HCCL_SIM_E_NOT_SUPPORT;
    }
    
    auto pusher = GetPusher(0);
    if (!pusher) {
        return HCCL_SIM_E_NOT_FOUND;
    }
    return pusher->Push((const char*)&req, sizeof(HcclTaskReq));
}

HcclVmResult HcclIpcComm::Pull(HcclTaskReq& req)
{
    if (!m_mode) {
        return HCCL_SIM_E_NOT_SUPPORT;
    }
    auto puller = GetPuller(0);
    if (!puller) {
        return HCCL_SIM_E_NOT_FOUND;
    }
    size_t reqLen = sizeof(HcclTaskReq);
    return puller->Pull((char*)&req, reqLen);
}

HcclVmResult HcclIpcComm::Push(const HcclTaskRsp& rsp)
{
    if (!m_mode) {
        return HCCL_SIM_E_NOT_SUPPORT;
    }

    auto pusher = GetPusher(GetRankIdFromTaskCid(rsp.taskCid));
    if (!pusher) {
        return HCCL_SIM_E_NOT_FOUND;
    }
    return pusher->Push((const char*)&rsp, sizeof(HcclTaskRsp));
}

HcclVmResult HcclIpcComm::Pull(uint16_t rankId, HcclTaskRsp& rsp)
{
    if (m_mode) {
        return HCCL_SIM_E_NOT_SUPPORT;
    }

    auto puller = GetPuller(rankId);
    if (!puller) {
        return HCCL_SIM_E_NOT_FOUND;
    }
    size_t rspLen = sizeof(HcclTaskRsp);
    return puller->Pull((char*)&rsp, rspLen);
}

HcclIpcClient* HcclIpcComm::GetPusher(uint16_t qId)
{
    std::string pusherAddr;
    if (m_mode) {   // runtime
        pusherAddr = "proxy_" + std::to_string(qId);
    } else {    // proxy
        pusherAddr = "runtime_" + std::to_string(qId);
    }

    auto iter = m_pushers.find(pusherAddr);
    if (iter != m_pushers.end()) {
        return iter->second;
    }
    auto pusher = HcclIpcClientFactory::GetInstance().CreateObject(TYPE_SHM);
    if (pusher == nullptr) {
        return nullptr;
    }
    pusher->Connect(pusherAddr);

    m_pushers[pusherAddr] = pusher;
    
    return pusher;
}

HcclIpcServer* HcclIpcComm::GetPuller(uint16_t qId)
{
    if (m_puller) {
        return m_puller;
    }

    std::string pullerAddr;
    if (m_mode) {   // runtime
        pullerAddr = "runtime_" + std::to_string(qId);
    } else {    // proxy
        pullerAddr = "proxy_" + std::to_string(qId);
    }

    m_puller = HcclIpcServerFactory::GetInstance().CreateObject(TYPE_SHM);
    if (m_puller) {
        m_puller->Bind(pullerAddr);
    }
    return m_puller;
}

