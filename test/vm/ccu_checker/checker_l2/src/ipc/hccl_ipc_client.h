#ifndef HCCL_IPC_CLIENT_H
#define HCCL_IPC_CLIENT_H

#include <string>
#include "hccl_common_defs.h"

class HcclIpcClient {
public:
    HcclIpcClient() {}
    virtual ~HcclIpcClient() {}
    virtual HcclSim::HcclVmResult Connect(std::string &addr) = 0;
    virtual HcclSim::HcclVmResult Push(const char *msg, const size_t size) = 0;
};

#endif