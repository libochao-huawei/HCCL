#ifndef HCCL_IPC_SERVER_H
#define HCCL_IPC_SERVER_H

#include <string>
#include "hccl_common_defs.h"

class HcclIpcServer {
public:
    HcclIpcServer() {}
    virtual ~HcclIpcServer() {}
    virtual HcclSim::HcclVmResult Bind(std::string& addr) = 0;
    virtual HcclSim::HcclVmResult Pull(char* recvBuf, size_t& recvSize) = 0;
};

#endif