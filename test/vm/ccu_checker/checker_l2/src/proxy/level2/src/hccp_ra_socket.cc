#include <unistd.h>
#include <vector>
#include <atomic>
#include <stdio.h>
#include <pthread.h>
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "runtime/base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "hccl_sim_shm_manager.h"
#include "task_status_cache.h"
// #include "hccl_vm.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "sim_runner_ops.h"
#include "hccp_common.h"
#include "ip_address.h"
#include "sim_runner_common.h"
#include "hccl_vm_log.h"

extern uint64_t g_cur_server_key;

constexpr uint32_t RA_SOCKET_BUF_SIZE = (64 * 1024);

#define MAKE_FD(client, server) ((client << 32) | server)
#define FD_LOCAL(fd_) (fd_ >> 32)
#define FD_PEER(fd_) (fd_ & 0x00000000FFFFFFFF)

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static void GenRaSocketBufKey(uint64_t localId, uint64_t peerId, std::vector<std::string> &keys)
{
    std::string key0 = "ra_socket_" + std::to_string(localId) + "_to_" + std::to_string(peerId);
    std::string key1 = "ra_socket_" + std::to_string(peerId) + "_to_" + std::to_string(localId);
    keys.push_back(key0);
    keys.push_back(key1);
}

static void GetRaSendSocketkeyByFd(uint64_t socketFd, std::string &key)
{
    key = "ra_socket_" + std::to_string(FD_LOCAL(socketFd)) + "_to_" + std::to_string(FD_PEER(socketFd));
}

static void GetRaRecvSocketkeyByFd(uint64_t socketFd, std::string &key)
{
    key = "ra_socket_" + std::to_string(FD_PEER(socketFd)) + "_to_" + std::to_string(FD_LOCAL(socketFd));
}

struct RaSocketBuff {
    ipc::interprocess_mutex mutex{};
    char data[RA_SOCKET_BUF_SIZE];
    uint64_t size{0};
    uint32_t sendBytes{0};
    uint32_t recvBytes{0};
};

int RaSocketInit(int mode, struct rdev rdevInfo, void **socketHandle)
{
    sim::Device device{};
    if (GetDeviceByPhysicalId(rdevInfo.phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("[RASOCKET] get device by phy id {} failed.", rdevInfo.phyId);
        return -1;
    }

    Hccl::BinaryAddr ba{};
    memcpy(&ba, &rdevInfo.localIp, sizeof(Hccl::BinaryAddr));
    auto ipAddr = Hccl::IpAddress(ba, AF_INET6).GetIpStr().substr(2);
    
    sim::Port port{};
    if (GetPortByIpAddr(ipAddr, port) != 0) {
        HCCL_VM_ERROR("[RASOCKET] cannot find remote ip {} ", ipAddr);
        return -1;
    }

    auto deviceIdx = device.id;
    auto ipIdx = port.id;

    auto ret = RunnerDB::GetOneByPred<sim::RaSocket>(
        [deviceIdx, ipIdx](const sim::RaSocket &so) { return (so.device_id == deviceIdx && so.ip_id == ipIdx); });
    if (ret.second) {
        *socketHandle = (void *)(uintptr_t)ret.first.id;
        HCCL_VM_INFO("[RASOCKET] Found socket socketFd:{:d}",ret.first.id);
        return 0;
    }

    sim::RaSocket socket{};
    socket.device_id = deviceIdx;
    socket.ip_id = port.id;
    auto socketfd = RunnerDB::Add<sim::RaSocket>(socket);
    *socketHandle = (void *)(uintptr_t)socketfd;

    printf("[zhupc][%lu][%lu][%s][%d] add socketHandle:%lu\n", (uint64_t)getpid(), pthread_self(), __func__, __LINE__, socketfd);
    fflush(stdout);
    return 0;
}

int RaSocketInitV1(int mode, struct SocketInitInfoT socketInit, void **socketHandle)
{
    struct rdev rdevInfo {};
    return RaSocketInit(mode, rdevInfo, socketHandle);
}

int RaSocketDeinit(void *socketHandle)
{
    uint64_t localFd = (uint64_t)(uintptr_t)(socketHandle);
    RunnerDB::Delete<sim::RaSocket>(localFd);

    printf("[zhupc][%lu][%lu][%s][%d] delete socketHandle:%lu\n",
        (uint64_t)getpid(),
        pthread_self(),
        __func__,
        __LINE__,
        localFd);
    fflush(stdout);
    return 0;
}

int RaSocketListenStart(struct SocketListenInfoT conn[], uint32_t num)
{
    for (uint32_t i = 0; i < num; i++) {
        uint64_t socketHandle = (uint64_t)(uintptr_t)(conn[i].socketHandle);
        auto socketRes = RunnerDB::GetById<sim::RaSocket>(socketHandle);
        if (!socketRes.has_value()) {
            HCCL_VM_ERROR("[{}] can not get Socket:{:d}", __func__, socketHandle);
            return -1;
        }

        // 更新角色
        RunnerDB::Update<sim::RaSocket>(socketHandle, [](sim::RaSocket &sock) { sock.state = 1; });
        HCCL_VM_INFO("[{}] socket {:d} is a server", __func__, socketHandle);
    }

    return 0;
}

int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num)
{
    return 0;
}

int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num)
{
    printf("[zhupc][%lu][%lu][%s][%d] enter num:%u\n", (uint64_t)getpid(), pthread_self(), __func__,__LINE__, num);
    fflush(stdout);
    for (int i = 0; i < num; i++) {
        uint64_t socketHandle = (uint64_t)(uintptr_t)(conn[i].socketHandle);
        auto socketRes = RunnerDB::GetById<sim::RaSocket>(socketHandle);
        if (!socketRes.has_value()) {
            HCCL_VM_ERROR("[{}] can not get Local Ra Socket:{:d}", __func__, socketHandle);
            return -1;
        }

        // 更新角色
        RunnerDB::Update<sim::RaSocket>(socketHandle, [](sim::RaSocket &sock) {
            sock.role = 1;
            sock.state = 2;
        });
        HCCL_VM_INFO("[{}] socket {:d} is a client", __func__, socketHandle);
        Hccl::BinaryAddr ba;
        memcpy(&ba, &conn[i].remoteIp, sizeof(Hccl::BinaryAddr));
        auto ipAddr = Hccl::IpAddress(ba, AF_INET6).GetIpStr().substr(2);
        sim::Port port{};
        if (GetPortByIpAddr(ipAddr, port) != 0) {
            HCCL_VM_ERROR("[{}] cannot find remote ip addr:{:d}", __func__, conn[i].remoteIp.addr.s_addr);
            return -1;
        }

        auto device_id = port.device_id;
        auto ip_id = port.id;
        HCCL_VM_INFO("[{}] IP INFO: ip= {}, serverIdx={:d}", __func__, ipAddr, g_cur_server_key);
        uint32_t count = 0;
        while (true) {
            if (count++ >= 5) {
                break;
            }
            auto socketRes = RunnerDB::GetOneByPred<sim::RaSocket>(
                [device_id, ip_id](const sim::RaSocket &socket) { return ((socket.device_id == device_id) && (socket.ip_id == ip_id)); });
            if (!socketRes.second) {
                HCCL_VM_WARN("[{}] cannot find remote Socket:{:d}, try again", __func__, device_id);
                sleep(1);
                continue;
            }

            std::vector<std::string> buffPairKey;
            GenRaSocketBufKey(socketHandle, socketRes.first.id, buffPairKey);
            for (auto &key : buffPairKey) {
                printf("[zhupc][%lu][%lu][%s][%d] local:%lu, peer:%lu, create key:%s\n", (uint64_t)getpid(), pthread_self(), __func__, __LINE__, socketHandle, socketRes.first.id, key.data());
                fflush(stdout);
                SHMManager::ConstructShmObject<RaSocketBuff>(key);
            }

            sim::RaSocketPair socketpair{};
            socketpair.client_id = socketHandle;
            socketpair.server_id = socketRes.first.id;
            socketpair.ref_cnt = 0;
            RunnerDB::Add<sim::RaSocketPair>(socketpair);
            break;
        }
    }
    return 0;
}

/*
RS_CONN_ROLE_SERVER = 0,
RS_CONN_ROLE_CLIENT = 1,
*/
int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int *connectedNum)
{
    HCCL_VM_INFO("[zhupc][{}] num:{:d}", __func__, num);
    int connected = 0;
    for (int i = 0; i < num; i++) {
        uint64_t socketHandle = (uint64_t)(uintptr_t)(conn[i].socketHandle);
        auto socketRes = RunnerDB::GetById<sim::RaSocket>(socketHandle);
        if (!socketRes.has_value()) {
            HCCL_VM_ERROR("[{}] can not get Local Ra Socket:{:d}", __func__, socketHandle);
            return -1;
        }

        uint32_t count = 0;
        while (true) {
            if (count++ >= 5) {
                break;
            }

            Hccl::BinaryAddr ba;
            memcpy(&ba, &conn[i].remoteIp, sizeof(Hccl::BinaryAddr));
            auto ipAddr = Hccl::IpAddress(ba, AF_INET6).GetIpStr().substr(2);
            sim::Port port{};
            if (GetPortByIpAddr(ipAddr, port) != 0) {
                HCCL_VM_ERROR("[{}] cannot find remote ip addr:{:d}", __func__, conn[i].remoteIp.addr.s_addr);
                return 0;
            }

            auto device_id = port.device_id;
            auto ip_id = port.id;
            auto socketRes = RunnerDB::GetOneByPred<sim::RaSocket>(
                [device_id, ip_id](const sim::RaSocket &socket) { return ((socket.device_id == device_id) && (socket.ip_id == ip_id)); });
            if (!socketRes.second) {
                HCCL_VM_WARN("[{}] cannot remote Socket: role= {:d}, devId={:d} sleep 1", __func__, role, device_id);
                sleep(1);
                continue;
            }

            auto peerSocketHandle = socketRes.first.id;

            auto socketPairRes = RunnerDB::GetOneByPred<sim::RaSocketPair>(
                [socketHandle, peerSocketHandle](const sim::RaSocketPair &pair) {
                    return (((pair.client_id == socketHandle) && (pair.server_id == peerSocketHandle)) ||
                           ((pair.client_id == peerSocketHandle) && (pair.server_id == socketHandle)));
                });
            if (!socketPairRes.second) {
                HCCL_VM_WARN(
                    "[{}] cannot pair Socket local:{:d} to peer:{:d} sleep 1", __func__, socketHandle, peerSocketHandle);
                sleep(1);
                continue;
            }

            auto pairId = socketPairRes.first.id;
            RunnerDB::Update<sim::RaSocketPair>(pairId, [](sim::RaSocketPair &pair) { pair.ref_cnt += 1; });

            conn[i].fdHandle = (void *)(uintptr_t)(MAKE_FD(socketHandle, peerSocketHandle));
            printf("[zhupc][%lu][%lu][%s][%d] local:%lu, peer:%lu get fd ref:%u \n", (uint64_t)getpid(), pthread_self(), __func__,__LINE__, socketHandle, peerSocketHandle, socketPairRes.first.ref_cnt + 1);
            fflush(stdout);
            conn[i].status = 1;
            *connectedNum += 1;
            break;
        }
    }
    return 0;
}

int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num)
{
    for (int i = 0; i < num; i++) {
        uint64_t socketFd = (uint64_t)(uintptr_t)(conn[i].fdHandle);
        auto localFd = FD_LOCAL(socketFd);
        auto peerFd = FD_PEER(socketFd);

        auto socketPairRes =
            RunnerDB::GetOneByPred<sim::RaSocketPair>([localFd, peerFd](const sim::RaSocketPair &pair) {
                return (((pair.client_id == localFd) && (pair.server_id == peerFd)) ||
                           ((pair.client_id == peerFd) && (pair.server_id == localFd)));
            });
        if (!socketPairRes.second) {
            HCCL_VM_ERROR(
                "[{}] cannot get pair Socket local:{:d} to peer:{:d} sleep 1", __func__, localFd, peerFd);
            continue;
        }

        if (socketPairRes.first.ref_cnt > 1) {
            auto pairId = socketPairRes.first.id;
            RunnerDB::Update<sim::RaSocketPair>(pairId, [](sim::RaSocketPair &pair) { pair.ref_cnt -= 1; });
            HCCL_VM_INFO("[zhupc]local:{:d}, peer:{:d} ref cnt:{:d}", localFd, peerFd, socketPairRes.first.ref_cnt - 1);
            fflush(stdout);
        } else {
            RunnerDB::Delete<sim::RaSocketPair>(socketPairRes.first.id);
            std::vector<std::string> buffPairKey;
            GenRaSocketBufKey(localFd, peerFd, buffPairKey);
            printf("[zhupc][%lu][%lu][%s][%d] local:%lu, peer:%lu clsoe key:%s\n", (uint64_t)getpid(), pthread_self(), __func__,__LINE__, localFd, peerFd, buffPairKey[0].data());
            fflush(stdout);
            for (auto &key : buffPairKey) {
                SHMManager::DestroyShmObject<RaSocketBuff>(key);
            }
        }
    }
    return 0;
}

int RaSocketBatchAbort(struct SocketConnectInfoT conn[], unsigned int num)
{
    return 0;
}

int RaSocketSend(const void *fdHandle, const void *data, unsigned long long size, unsigned long long *sentSize)
{
    clock_t start = clock();
    uint64_t socketFd = (uint64_t)(uintptr_t)(fdHandle);
    std::string socketKey;
    GetRaSendSocketkeyByFd(socketFd, socketKey);

    RaSocketBuff *buf = SHMManager::FindShmObject<RaSocketBuff>(socketKey);
    if (buf == nullptr) {
        HCCL_VM_ERROR("[{}] cannot pair Socket client:{:d} to server:{:d}, key={}",
            __func__,
            FD_LOCAL(socketFd),
            FD_PEER(socketFd),
            socketKey);
        return 1;
    }
    clock_t end1 = clock();
    double time1 = double(end1 - start) / CLOCKS_PER_SEC;

    ipc::scoped_lock<ipc::interprocess_mutex> lock(buf->mutex);

    if ((buf->size + size > RA_SOCKET_BUF_SIZE)) {
        HCCL_VM_ERROR("[{}] Socket client:{:d} to server:{:d} send to big {:d}",
            __func__,
            FD_LOCAL(socketFd),
            FD_PEER(socketFd),
            size);
        return 1;
    }

    memcpy(buf->data + buf->size, data, size);
    buf->size += size;
    *sentSize = size;

    buf->sendBytes += size;

    printf("[zhupc][%lu][%lu][%s][%d] local:%lu, peer:%lu send key:%s Send bytes:%u\n", (uint64_t)getpid(), pthread_self(), __func__, __LINE__, FD_LOCAL(socketFd), FD_PEER(socketFd), socketKey.data(), buf->sendBytes);
    fflush(stdout);
    clock_t end2 = clock();
    double time2 = double(end2 - end1) / CLOCKS_PER_SEC;
    return 0;
}

int RaSocketRecv(const void *fdHandle, void *data, unsigned long long size, unsigned long long *receivedSize)
{
    clock_t start = clock();
    uint64_t socketFd = (uint64_t)(uintptr_t)(fdHandle);
    std::string socketKey;
    GetRaRecvSocketkeyByFd(socketFd, socketKey);

    RaSocketBuff *buf = SHMManager::FindShmObject<RaSocketBuff>(socketKey);
    if (buf == nullptr) {
        HCCL_VM_ERROR(
            "[{}] cannot pair Socket client:{:d} to server:{:d}", __func__, FD_LOCAL(socketFd), FD_PEER(socketFd));
        return -1;
    }
    clock_t end1 = clock();
    double time1 = double(end1 - start) / CLOCKS_PER_SEC;

    int retryCnt = 0;
    while (retryCnt++ <= 5) {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(buf->mutex);
        if (buf->size == 0) {
            HCCL_VM_WARN("[{}] return socket eagain.", __func__);
            sleep(1);
            continue;
        }

        // *receivedSize = (size < buf->size) ? size : buf->size;
        *receivedSize = size;
        buf->recvBytes += size;
        memcpy(data, buf->data, size);
        buf->size -= size;
        if (buf->size > 0) {
            memmove(buf->data, buf->data + size, buf->size);
        }
        printf("[zhupc][%lu][%lu][%s][%d] local:%lu, peer:%lu recv key:%s Recv bytes:%u\n", (uint64_t)getpid(), pthread_self(), __func__, __LINE__, FD_LOCAL(socketFd), FD_PEER(socketFd), socketKey.data(), buf->recvBytes);
        fflush(stdout);
        break;
    }
    clock_t end2 = clock();
    double time2 = double(end2 - end1) / CLOCKS_PER_SEC;
    return 0;
}

int RaEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event)
{
    return 0;
}

int RaEpollCtlMod(const void *fdHandle, enum RaEpollEvent event)
{
    return 0;
}

int RaEpollCtlDel(const void *fdHandle)
{
    return 0;
}

int RaSetTcpRecvCallback(const void *socketHandle, const void *callback)
{
    return 0;
}

/////////////////////////////////async/////////////////////////////
int RaGetAsyncReqResult(void *reqHandle, int *reqResult)
{
    return 0;
}

int RaSocketBatchConnectAsync(struct SocketConnectInfoT conn[], unsigned int num, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return RaSocketBatchConnect(conn, num);
}

int RaSocketListenStartAsync(struct SocketListenInfoT conn[], unsigned int num, void **reqHandle)
{
    return -1;
}

int RaSocketListenStopAsync(struct SocketListenInfoT conn[], unsigned int num, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int RaSocketBatchCloseAsync(struct SocketCloseInfoT conn[], unsigned int num, void **reqHandle)
{
    return RaSocketBatchClose(conn, 0);
}

int RaSocketSendAsync(
    const void *fdHandle, const void *data, unsigned long long size, unsigned long long *sentSize, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return RaSocketSend(fdHandle, data, size, sentSize);
}

int RaSocketRecvAsync(
    const void *fdHandle, void *data, unsigned long long size, unsigned long long *receivedSize, void **reqHandle)
{
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    clock_t start = clock();
    uint64_t socketFd = (uint64_t)(uintptr_t)(fdHandle);
    std::string socketKey;
    GetRaRecvSocketkeyByFd(socketFd, socketKey);
    std::string newSocketKey;
    GetRaSendSocketkeyByFd(socketFd, newSocketKey);

    RaSocketBuff *buf = SHMManager::FindShmObject<RaSocketBuff>(socketKey);
    if (buf == nullptr) {
        HCCL_VM_ERROR(
            "[{}] cannot pair Socket client:{:d} to server:{:d}", __func__, FD_LOCAL(socketFd), FD_PEER(socketFd));
        return -1;
    }
    clock_t end1 = clock();
    double time1 = double(end1 - start) / CLOCKS_PER_SEC;

    ipc::scoped_lock<ipc::interprocess_mutex> lock(buf->mutex);
    if (buf->size == 0) {
        sleep(1);
        HCCL_VM_WARN("[{}] return socket eagain.key = {}", __func__, socketKey);
        return 0;
    }

    *receivedSize = (size < buf->size) ? size : buf->size;
    // *receivedSize = size;
    buf->recvBytes += *receivedSize;
    memcpy(data, buf->data, *receivedSize);
    buf->size -= *receivedSize;
    if (buf->size > 0) {
        memmove(buf->data, buf->data + *receivedSize, buf->size);
    }
    printf("[zhupc][%lu][%lu][%s][%d] local:%lu, peer:%lu ori key:%s - new key:%s recv bytes:%u, Recv Size=%llu\n",
        (uint64_t)getpid(),
        pthread_self(),
        __func__,
        __LINE__,
        FD_LOCAL(socketFd),
        FD_PEER(socketFd),
        socketKey.data(),
        newSocketKey.data(),
        buf->recvBytes,
        *receivedSize);
    fflush(stdout);
    clock_t end2 = clock();
    double time2 = double(end2 - end1) / CLOCKS_PER_SEC;
    return 0;
}

#ifdef __cplusplus
}
#endif  // __cplusplus