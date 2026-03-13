/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: fake socket stub header
 */

#include <cstring>
#include <set>
#include "fake_socket.h"
#include <iostream>
#include "SimRunnerMgr.h"
#include "CcuResourceManager.h"

SocketFd* FakeSocket::GetSocketFdForSR(int fd, bool isSend)
{
    int actFd = fd;
    if (!isSend) {
        actFd = MAKE_FD(FD_PEER_RANK(fd), FD_LOCAL_RANK(fd));
    }
    int pos = FindFdPos(actFd);
    if (pos == -1) {
        return nullptr;
    }
    return &(shmPub_->socket.socketFds[pos]);
}

bool FakeSocket::Send(int *fdHandle, const void *data, unsigned long long int size, unsigned long long *sent_size)
{
    ShmPoolLock shmPoolLock;

    SocketFd *pSckFd = GetSocketFdForSR(*fdHandle, true);
    if ((pSckFd == nullptr) || (pSckFd->totalLen + size > DEVICE_SOCKET_SEND_MAX)) {
        std::cout << " FakeSocket::Send failed! size = " << size << std::endl;
        return false;  // 底层限速，待实现，遇到了先报错
    }

    *sent_size = size;
    memcpy(pSckFd->sendBuff + pSckFd->totalLen, data, size);
    pSckFd->totalLen += size;
    return true;
}

bool FakeSocket::Recv(int *fdHandle, void *data, unsigned long long int size, unsigned long long *received_size)
{
    ShmPoolLock shmPoolLock;

    SocketFd *pSckFd = GetSocketFdForSR(*fdHandle, false);
    if ((pSckFd == nullptr) || (size > DEVICE_SOCKET_SEND_MAX) || (pSckFd->totalLen < size)) {
        return false;
    }

    *received_size = size;
    memcpy(data, pSckFd->sendBuff, size);
    memmove(pSckFd->sendBuff, pSckFd->sendBuff + size, pSckFd->totalLen - size);
    pSckFd->totalLen -= size;
    return true;
}

bool FakeSocket::Connect(struct socket_connect_info_t &conn)
{
    ShmPoolLock shmPoolLock;

    auto rmtIpDec = conn.remote_ip.addr.s_addr;
    NetDeviceInfo devInfo;
    auto ret = SimRunnerMgr::GetInstance().GetNetDeviceMgr()->GetRmtDevInfo(rmtIpDec, devInfo);
    if (!ret) {
        std::cout<<"[FakeSocket][Connect][ERROR] Get remote device info failed. ipDec="<<rmtIpDec<<std::endl;
        return false;
    }
    int fd = MAKE_FD(rankId_, devInfo.rankId);
    if (FindFdPos(fd) != -1) {
        return true;
    }

    InsertFdPos(fd);
    return true;
}

bool FakeSocket::Get(int role, struct socket_info_t &conn)
{
    ShmPoolLock shmPoolLock;

    auto rmtIpDec = conn.remote_ip.addr.s_addr;
    NetDeviceInfo devInfo;
    auto ret = SimRunnerMgr::GetInstance().GetNetDeviceMgr()->GetRmtDevInfo(rmtIpDec, devInfo);
    if (!ret) {
        std::cout<<"[FakeSocket][Get][ERROR] Get remote device info failed. ipDec="<<rmtIpDec<<std::endl;
        return false;
    }
    int fd = MAKE_FD(rankId_, devInfo.rankId);
    int pos = FindFdPos(fd);
    if (pos == -1) {
        //    SOCKET_ROLE_SERVER = 0,          /* server 角色 */
        //    SOCKET_ROLE_CLIENT = 1,          /* client 角色 */
        if (role == 1) {
            return true;
        } else {
            // 对于server role, 并没有调用connect。 需要在此处单独处理
            pos = InsertFdPos(fd);
        }
    }
    if (pos == -1) {
        std::cout<<"[FakeSocket][Get][ERROR] Insert fd pos failed. ipDec="<<rmtIpDec<<std::endl;
        return false;
    }

    conn.status = 1;
    conn.fd_handle = &(shmPub_->socket.socketFds[pos].fd);
    return true;
}

int FakeSocket::FindFdPos(int fd)
{
    for (int i = 0; i < MAX_STREAM_NUM; i++) {
        if (shmPub_->socket.socketFds[i].fd == fd) {
            return i;
        }
    }

    return -1; // not found
}

int FakeSocket::InsertFdPos(int fd)
{
    for (int i = 0; i < MAX_STREAM_NUM; i++) {
        if (shmPub_->socket.socketFds[i].fd == -1) {
            shmPub_->socket.socketFds[i].fd = fd;
            shmPub_->socket.socketFds[i].totalLen = 0;
            return i;
        }
    }

    return -1; // failed
}

int *FakeSocket::GetSocketHandle()
{
    return &(shmPub_->socket.socketHandleStore[rankId_]);
}

void FakeSocket::SetDeviceId(int rankId)
{
    rankId_ = rankId;
}

void NetDeviceMgr::SetDeviceId(int rankId)
{
    rankId_ = rankId;
}

bool NetDeviceMgr::InitLocalId2RankIdMap()
{
    string filepath  = "./ranktable.json";
    std::ifstream file(filepath.c_str());
    if (!file.is_open()) {
        std::cout << "Failed to open file ranktable" << std::endl;
        return false;
    }
    nlohmann::json json;
    file >> json;
    try {
        nlohmann::json ranklist = json.at("rank_list");
        for (auto& rank : ranklist) {
            int localId = rank.at("local_id");
            int rankId = rank.at("rank_id");
            localId2RankIdMap_.insert(std::make_pair(localId, rankId));
            rankId2LocalIdMap.insert(std::make_pair(rankId, localId));
        }
    } catch (const nlohmann::json::out_of_range& e) {
        std::cout << "[InitLocalId2RankIdMap][ERROR] Key not found" << std::endl;
        return false;
    }
    return true;
}

bool NetDeviceMgr::ParseRankTable(std::set<int> &rankTableLocalIds)
{
    string filepath  = "./ranktable.json";
    std::ifstream file(filepath.c_str());
    if (!file.is_open()) {
        std::cout << "Failed to open file ranktable.json" << std::endl;
        return false;
    }
    nlohmann::json json;
    file >> json;

    try {
        nlohmann::json rankList = json.at("rank_list");
        for (auto& rank : rankList) {
            int localId = rank.at("local_id");
            rankTableLocalIds.insert(localId);
        }
    } catch (const nlohmann::json::out_of_range& e) {
        std::cout << "[ParseRankTable][ERROR] Key not found" << std::endl;
        return false;
    }
    return true;
}

bool NetDeviceMgr::ParseDieInfo(const std::set<int> &rankTableLocalIds)
{
    string filepath  = "./die_info.json";
    std::ifstream file(filepath.c_str());
    if (!file.is_open()) {
        std::cout << "Failed to open file die_info" << std::endl;
        return false;
    }
    nlohmann::json json;
    file >> json;

    uint64_t rdmaHandleBase = rdmaHandleBase_ + 1;
    ip2DevInfoMap_.resize(rankSize_);
    try {
        nlohmann::json edgelist = json.at("edge_list");
        for (auto& rank : edgelist) {
            int localId = rank.at("local_id");
            if (rankTableLocalIds.count(localId) == 0) {
                continue;
            }
            int rankId = GetRankIdByLocalId(localId);
            if (rankId == INT32_MAX) {
                std::cout<<"[InitRandkId2DieMap][ERROR] Get rank id by local id["<< localId<<"], failed"<<std::endl;
                return false;
            }
            int tmp_func = 0;
            nlohmann::json dieList = rank.at("die_info");
            std::map<u64, struct NetDeviceInfo> rankDieInfo;
            for (auto& die : dieList) {
                int dieId = die.at("die_id");
                std::string ip = die.at("ip");
                u64 ipDec = IpAddress2Dec(ReverseIpAddress(ip));
                struct NetDeviceInfo  tmp = {localId, rankId, dieId, tmp_func++, rdmaHandleBase++};
                HCCL_INFO("InitRandkId2DieMap:ip[%s] localId[%d] rankId[%d] dieId[%d] funcId[%d]", ip.c_str(), tmp.localId, tmp.rankId, tmp.dieId, tmp.funcId);
                rankDieInfo.insert(std::make_pair(ipDec, tmp));
            }
            ip2DevInfoMap_[rankId] = rankDieInfo;
        }
    } catch (const nlohmann::json::out_of_range& e) {
        std::cout << "[InitRandkId2DieMap][ERROR] Key not found" << std::endl;
        return false;
    }
    return true;
}

bool NetDeviceMgr::InitRandkId2DieMap()
{
    std::set<int> rankTableLocalIds;
    if (ParseRankTable(rankTableLocalIds) == false) {
        return false;
    }

    if (ParseDieInfo(rankTableLocalIds) == false) {
        return false;
    }

    DumpIp2DevInfoMap();
    return true;
}

NetDeviceMgr::NetDeviceMgr(int rankSize, uint32_t ccuFlag) : rankSize_(rankSize), ccuFlag_(ccuFlag)
{
    HcclResult ret = hrtGetDeviceType(devType_);
    if (ret == HCCL_SUCCESS && devType_ == DevType::DEV_TYPE_910_95) {
        InitLocalId2RankIdMap();
        InitRandkId2DieMap();
    }
}

int NetDeviceMgr::GetIpNum(int rankId)
{
    return ip2DevInfoMap_[rankId].size();
}

std::map<u64, struct NetDeviceInfo> NetDeviceMgr::GetDieInfoMap(int phyDevId)
{
    return ip2DevInfoMap_[phyDevId];
}

bool NetDeviceMgr::GetRmtDevInfo(uint64_t ipDec, NetDeviceInfo &devInfo)
{
    if (devType_ != DevType::DEV_TYPE_910_95) {
        devInfo.rankId = ipDec;
        return true;
    }
    for (const auto &rankDieInfo : ip2DevInfoMap_) {
        auto dieRes = rankDieInfo.find(ipDec);
        if (dieRes != rankDieInfo.end()) {
            devInfo = dieRes->second;
            return true;
        }
    }
    return false;
}

void NetDeviceMgr::DumpIp2DevInfoMap()
{
    for (const auto &rankDieInfo : ip2DevInfoMap_) {
        for (const auto &dieInfo : rankDieInfo) {
            std::cout << "[DumpIp2DevInfoMap] ipDec=" <<std::left <<std::setw(10)<<std::hex << dieInfo.first << ", localId=" << dieInfo.second.localId
                      << ", rankId=" << dieInfo.second.rankId << ", dieId=" << dieInfo.second.dieId
                      << ", funcId=" << dieInfo.second.funcId << ", rdmaHandle=" <<std::left <<std::setw(10) <<std::hex << dieInfo.second.rdmaHandle
                      << std::endl;
        }
    }
}

bool NetDeviceMgr::GetRmtDevInfo(void *ctx_handle, NetDeviceInfo &devInfo)
{
    uint64_t ipDec = GetIpDec(ctx_handle);   // 根据ctx_handle到ip的映射表获取ip
    return GetRmtDevInfo(ipDec, devInfo);
}

std::string NetDeviceMgr::ReverseIpAddress(const std::string &ip)
{
    std::vector<int> parts;
    std::stringstream ss(ip);
    std::string part;

    // 分割字符串
    while (std::getline(ss, part, '.')) {
        parts.push_back(std::stoi(part));
    }

    // 反转每个部分
    std::reverse(parts.begin(), parts.end());

    // 重新组合
    std::stringstream result;
    for (size_t i = 0; i < parts.size(); ++i) {
        result << parts[i];
        if (i != parts.size() - 1) {
            result << ".";
        }
    }

    return result.str();
}

u64 NetDeviceMgr::IpAddress2Dec(const std::string &ip)
{
    std::istringstream iss(ip);
    std::string byte;
    u64 result = 0;

    // 解析每个字节并转换为整数
    while (std::getline(iss, byte, '.')) {
        result = (result << 8) | std::stoi(byte);  // 将每个字节按位移位并与当前结果合并
    }

    return result;
}

int NetDeviceMgr::GetDevPhyId(int devLogicId)
{
    auto iter = rankId2LocalIdMap.find(devLogicId);
    if (iter != rankId2LocalIdMap.end()) {
        return iter->second;
    }

    return INT32_MAX;
}

int NetDeviceMgr::GetRankIdByLocalId(int localId)
{
    auto iter = localId2RankIdMap_.find(localId);
    if (iter != localId2RankIdMap_.end()) {
        return iter->second;
    }
    std::cout<<"[NetDeviceMgr][GetRankIdByLocalId] Get rank id by local id["<< localId<<"], failed. size="<<localId2RankIdMap_.size()<<std::endl;
    return INT32_MAX;
}

void NetDeviceMgr::SaveEid(int dieId, int chId, uint64_t ipDec, string ipAddr)
{
    int rmtDevId = 0XFFFF;
    int rmtDieId = 0XFFFF;
    if (ipAddr == "0.0.0.0") {
        if ((dieId == 0 && chId == 0) || (dieId == 1 && chId == 1)) {
            // die内环回：channel 0：die0 -> die0 或 channel 1: die1 -> die1
            rmtDevId = rankId_;
            rmtDieId = dieId;
        } else {
            // rank内die间环回：channel 1：die0 -> die1 或 channel 0: die1 -> die0
            rmtDevId = rankId_;
            rmtDieId = (dieId == 0 ? 1 : 0);
        }
        std::cout<<"[SaveEid][LoopChannel] chId="<<chId<<": local["<< rankId_<<":"<<dieId<<"] -> remote["<< rmtDevId<<":"<<rmtDieId<<"]"<<std::endl;
        CcuResouceManager::GetInstance().InitChannelId2RmtRankMap(rankId_, dieId, chId, rmtDevId, rmtDieId);
        return;
    }

    // rank间channel
    NetDeviceInfo rmtDevice;
    auto ret = GetRmtDevInfo(ipDec, rmtDevice);
    if (!ret) {
        std::cout<<"[NetDeviceMgr][SaveEid][ERROR] Get remote device info failed. ipDec="<<ipDec<<std::endl;
        return;
    }
    rmtDevId  = rmtDevice.rankId;
    rmtDieId  = rmtDevice.dieId;

    std::cout<<"[SaveEid][RmtChannel] chId="<<chId<<": local["<< rankId_<<":"<<dieId<<"] -> remote["<< rmtDevId<<":"<<rmtDieId<<"]"<<std::endl;
    CcuResouceManager::GetInstance().InitChannelId2RmtRankMap(rankId_, dieId, chId, rmtDevId, rmtDieId);
}

FakeUb::FakeUb()
{
    for (int i = 0; i < MAX_DEVICE_NUM_PER_SERVER; i++) {
        qpHandleStore_[i] = i;
    }
}

int *FakeUb::GetQpHandle()
{
    return &qpHandleStore_[rankId_];
}

int FakeUb::PushWqe(FakeWqe wqe)
{
    wqes_.push_back(wqe);
    return wqes_.size() - 1;
}

FakeWqe FakeUb::GetWqe(int idx)
{
    if (idx >= wqes_.size()) {
        FakeWqe tmp;
        return tmp;
    }

    return wqes_[idx];
}

void FakeUb::SetDeviceId(int rankId)
{
    rankId_ = rankId;
}

void NetDeviceMgr::SetHandle2AddrMap(void* ctx_handle, uint64_t ipDec)
{
    handle2AddrMap_.insert(std::make_pair(ctx_handle, ipDec));
}

uint64_t NetDeviceMgr::GetIpDec(void *ctx_handle)
{
    auto iter = handle2AddrMap_.find(ctx_handle);
    if (iter != handle2AddrMap_.end()) {
        return iter->second;
    }
    return 0;
}

void *NetDeviceMgr::GetRdmaHandle(uint64_t ipDec)
{
    for(auto iter : handle2AddrMap_){
        if (iter.second == ipDec) {
            return iter.first;
        }
    }
    std::cout <<  "NetDeviceMgr::GetRdmaHandle IpAddress not found" << std::endl;
    return nullptr;
}

int *NetDeviceMgr::GetRdmaHandleNew(uint64_t ipDec)
{
    if (ipDec == 0) {
        return reinterpret_cast<int*>(static_cast<uintptr_t>(rdmaHandleBase_));
    }
    int *rdmaHandle = nullptr;
    for (const auto &rankDieInfo : ip2DevInfoMap_) {
        auto dieRes = rankDieInfo.find(ipDec);
        if (dieRes != rankDieInfo.end()) {
            rdmaHandle = reinterpret_cast<int*>(static_cast<uintptr_t>(dieRes->second.rdmaHandle));
            break;
        }
    }
    return rdmaHandle;
}