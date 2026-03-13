#include "rdma_handle_manager.h"
#include "socket_handle_manager.h"
#include "rank_info_recorder.h"
#include "coll_gen_json.h"

namespace Hccl {
#define MAX_DEVICE_NUM 64

RdmaHandleManager::RdmaHandleManager()
{
    rdmaHandleMap.resize(MAX_DEVICE_NUM);
    for (u32 i = 0; i < rdmaHandleMap.size(); ++i) {
        rdmaHandleMap[i].resize(LINK_PROTO_TYPE_NUM);
    }
}

RdmaHandleManager::~RdmaHandleManager()
{
    ;
}

std::map<RdmaHandle, std::pair<uint32_t, uint32_t>> g_rdmaHandle2DieIdAndFuncId;
RdmaHandle g_rdmaHandlePtr = reinterpret_cast<RdmaHandle>(0x01);

RdmaHandle RdmaHandleManager::Create(u32 devPhyId, const PortData &localPort)
{
    RdmaHandle res = g_rdmaHandlePtr++;
    rdmaHandleMap[devPhyId][localPort.GetProto()][localPort.GetAddr()] = res;

    g_rdmaHandle2DieIdAndFuncId[res] = g_devId2Ip2DieIdAndFuncId[devPhyId][localPort.GetAddr()];
    return res;
}

RdmaHandle RdmaHandleManager::GetByIp(u32 devPhyId, const IpAddress &localIp)
{
    RdmaHandle res = rdmaHandleMap[devPhyId][LinkProtoType::UB][localIp];

    if (res == nullptr) {
        res = g_rdmaHandlePtr++;
        rdmaHandleMap[devPhyId][LinkProtoType::UB][localIp] = res;

        g_rdmaHandle2DieIdAndFuncId[res] = g_devId2Ip2DieIdAndFuncId[devPhyId][localIp];
        HCCL_INFO("Create one rdmahandle [%p], devPhyId [%u], ipAddr [%s]",
            res, devPhyId, localIp.Describe().c_str());
    }
    return res;
}

RdmaHandle RdmaHandleManager::Get(u32 devPhyId, const PortData &localPort)
{
    LinkProtoType localProto = localPort.GetProto();
    if (devPhyId > rdmaHandleMap.size() - 1 || localProto == LinkProtoType::HCCS_PCIE) {
        return nullptr;
    }

    IpAddress  localIp = localPort.GetAddr();
    RdmaHandle res     = rdmaHandleMap[devPhyId][localProto][localIp];
    if (res == nullptr) {
        if (localProto == LinkProtoType::RDMA) {
            res = Create(devPhyId, localPort);
        } else if (localProto == LinkProtoType::UB) {
            res = g_rdmaHandlePtr++;
            rdmaHandleMap[devPhyId][localProto][localIp] = res;

            g_rdmaHandle2DieIdAndFuncId[res] = g_devId2Ip2DieIdAndFuncId[devPhyId][localIp];
            HCCL_INFO("Create one rdmahandle [%p], devPhyId [%u], portAddr [%s]",
                res, devPhyId, localPort.GetAddr().Describe().c_str());
        }
    }

    return res;
}

std::pair<uint32_t, uint32_t> RdmaHandleManager::GetDieAndFuncId(RdmaHandle rdmaHandle)
{
    if (DieAndFuncIdMap.find(rdmaHandle) != DieAndFuncIdMap.end()) {
        return DieAndFuncIdMap[rdmaHandle];
    }

    DieAndFuncIdMap[rdmaHandle] = HraGetDieAndFuncId(rdmaHandle);
    return DieAndFuncIdMap[rdmaHandle];
}

JfcHandle RdmaHandleManager::GetJfcHandle(RdmaHandle rdmaHandle, HrtUbJfcMode jfcMode)
{
    return 1;
}

std::pair<TokenIdHandle, uint32_t> RdmaHandleManager::GetTokenIdInfo(RdmaHandle rdmaHandle, const BufferKey<uintptr_t, u64> &bufKey)
{
    return std::make_pair(0, 0);
}

RdmaHandleManager &RdmaHandleManager::GetInstance()
{
    static RdmaHandleManager rdmaHandleManager;
    return rdmaHandleManager;
}

void RdmaHandleManager::DestroyAll()
{
    DieAndFuncIdMap.clear();
    rdmaHandleMap.clear();
    rdmaHandleMap.resize(MAX_DEVICE_NUM);
    for (u32 i = 0; i < rdmaHandleMap.size(); ++i) {
        rdmaHandleMap[i].resize(LINK_PROTO_TYPE_NUM);
    }
}

}