#ifndef HCCL_IPC_COMM_H
#define HCCL_IPC_COMM_H

#include <string>
#include <map>
#include <memory>
#include "hccl_common_defs.h"
#include "hccl_ipc_client.h"
#include "hccl_ipc_server.h"

class HcclIpcComm {
public:
    using IpcClientMaps = std::map<std::string, HcclIpcClient*>;
    HcclIpcComm();
    ~HcclIpcComm();

    HcclSim::HcclVmResult Push(const HcclTaskReq& req);
    HcclSim::HcclVmResult Pull(HcclTaskReq& req);

    HcclSim::HcclVmResult Push(const HcclTaskRsp& rsp);
    HcclSim::HcclVmResult Pull(uint16_t rankId, HcclTaskRsp& rsp);
private:
    HcclIpcClient* GetPusher(uint16_t rankId);
    HcclIpcServer* GetPuller(uint16_t rankId);
private:
    HcclIpcServer*  m_puller;
    IpcClientMaps   m_pushers;
    int m_mode;// 0:proxy 1:virtual runtime
};

#endif