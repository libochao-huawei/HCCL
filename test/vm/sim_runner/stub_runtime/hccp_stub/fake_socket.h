/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: fake socket stub
 */

#ifndef HCCL_SIM_FAKE_SOCKET_H
#define HCCL_SIM_FAKE_SOCKET_H

#include <map>
#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "hccp.h"
#include "hccl_sim_pub_stub.h"
#include "ip_address.h"

using namespace std;

#define MAKE_FD(lcl_, peer_) ((lcl_ << 16) | peer_)
#define FD_LOCAL_RANK(fd_) (fd_ >> 16)
#define FD_PEER_RANK(fd_) (fd_ & 0x0000FFFF)

// WQE相关结构定义
enum WqeType {
    WRITE,
    WRITE_WITH_NOTIFY,
    REDUCE_WITH_NOTIFY,
};

struct FakeWqe {
    enum WqeType type;
    u64 notifyAddr;
    u64 localAddr;
    u64 remoteAddr;
    u32 size;
    u8 reduceOpType;
    u8 reduceDataType;
};

struct NetDeviceInfo {
    int localId;
    int rankId;
    int dieId; // David芯片支持，其他芯片为0
    int funcId;
    uint64_t rdmaHandle;
};

class FakeSocket {
public:
    FakeSocket(bool isAicpu, ShmPub *shmMgr) : aicpuFlag_(isAicpu), shmPub_(shmMgr) {};
    bool Connect(struct socket_connect_info_t &conn);
    bool Get(int role, struct socket_info_t &conn);
    bool Send(int *fdHandle, const void *data, unsigned long long size, unsigned long long *sent_size);
    bool Recv(int *fdHandle, void *data, unsigned long long size, unsigned long long *received_size);
    int *GetSocketHandle();
    void SetDeviceId(int rankId);

private:
    int FindFdPos(int fd);
    int InsertFdPos(int fd);
    SocketFd *GetSocketFdForSR(int fd, bool isSend);

private:
    int rankId_{0};
    bool aicpuFlag_{false};
    ShmPub *shmPub_{nullptr};
};

class FakeUb {
public:
    FakeUb();
    int *GetQpHandle();
    int PushWqe(FakeWqe wqe);
    FakeWqe GetWqe(int idx);
    void SetDeviceId(int rankId);

private:
    int rankId_{0};
    std::vector<FakeWqe> wqes_;
    int qpHandleStore_[MAX_DEVICE_NUM_PER_SERVER];
};

class NetDeviceMgr {
public:
    NetDeviceMgr(int rankSize, uint32_t ccuFlag);

    void SetDeviceId(int rankId);
    int GetDevPhyId(int devLogicId);
    void SetHandle2AddrMap(void *ctx_handle, uint64_t ipDec);
    void SaveEid(int dieId, int chId, uint64_t ipDec, string ipAddr);
    
    int GetIpNum(int rankId);
    int *GetRdmaHandleNew(uint64_t ipDec);
    void *GetRdmaHandle(uint64_t ipDec);
    bool GetRmtDevInfo(uint64_t ipDec, NetDeviceInfo &devInfo);
    bool GetRmtDevInfo(void *ctx_handle, NetDeviceInfo &devInfo);
    std::map<u64, struct NetDeviceInfo> GetDieInfoMap(int phyDevId);

private:
    bool InitLocalId2RankIdMap();
    bool InitRandkId2DieMap();
    u64 IpAddress2Dec(const std::string &ip);
    std::string ReverseIpAddress(const std::string &ip);
    uint64_t GetIpDec(void *ctx_handle);
    int GetRankIdByLocalId(int localId);
    void DumpIp2DevInfoMap();
    bool ParseRankTable(std::set<int> &rankTableLocalIds);
    bool ParseDieInfo(const std::set<int> &rankTableLocalIds);

private:
    int rankId_{0};
    int rankSize_{0};
    bool ccuFlag_{false};
    DevType devType_{DevType::DEV_TYPE_COUNT};
    uint64_t rdmaHandleBase_{0x10000000};
    std::map<int, int> localId2RankIdMap_;                   // 适配2D场景，localId -> rankId
    std::map<int, int> rankId2LocalIdMap;                    // 适配2D场景，将rankId映射为localId
    std::vector<std::map<u64, struct NetDeviceInfo>> ip2DevInfoMap_;  // David芯片专属：ip -> {rmtPhyDevId, dieId, funcId}
    std::map<void *, u64> handle2AddrMap_;                   // ccu handle -> ipaddr
};
#endif  // HCCL_SIM_FAKE_SOCKET_H
