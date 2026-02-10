#ifndef HCCL_IPC_FACTORY_H
#define HCCL_IPC_FACTORY_H

#include <vector>
#include <string>

#include "hccl_ipc_defs.h"
#include "hccl_ipc_server.h"
#include "hccl_ipc_client.h"

#include "hccl_ipc_shm_client.h"
#include "hccl_ipc_shm_server.h"

template <typename Factory, typename Object>
class HcclIpcFactory {
public:
    static Factory &GetInstance()
    {
        static Factory instance;
        return instance;
    }

protected:
    HcclIpcFactory() = default;
    ~HcclIpcFactory() = default;

    HcclIpcFactory(const HcclIpcFactory &) = delete;
    HcclIpcFactory(HcclIpcFactory &&) = delete;
    HcclIpcFactory &operator=(const HcclIpcFactory &) = delete;
    HcclIpcFactory &operator=(HcclIpcFactory &&) = delete;
};

class HcclIpcServerFactory : public HcclIpcFactory<HcclIpcServerFactory, HcclIpcServer> {
public:
    HcclIpcServer* CreateObject(HCCL_IPC_TYPE type)
    {
        switch (type) {
            case TYPE_SHM:
                return new HcclIpcShmServer();
            default:
                return nullptr;
        }
        return nullptr;
    }
};

class HcclIpcClientFactory : public HcclIpcFactory<HcclIpcClientFactory, HcclIpcClient> {
public:
    HcclIpcClient* CreateObject(HCCL_IPC_TYPE type)
    {
        switch (type) {
            case TYPE_SHM:
                return new HcclIpcShmClient();
            default:
                return nullptr;
        }
        return nullptr;
    }
};
#endif