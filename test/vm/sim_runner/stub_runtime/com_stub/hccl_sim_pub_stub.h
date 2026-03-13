/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim public stub header
 */

#ifndef HCCL_SIM_PUB_STUB_H
#define HCCL_SIM_PUB_STUB_H
#include <iostream>
#include <string>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <unistd.h>
#include "sal.h"
#include <sys/wait.h>
#include <signal.h>
#include <stdlib.h>
#include "../../virt_runtime_fwk/hccl_sim_situation.h"
#include <signal.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "../rts_stub/rts_stub.h"
#include "common/aicpu_hccl_def.h"
#undef HCCL_HDC_TYPE_D2H
#undef HCCL_HDC_TYPE_H2D
#include "ccuMicrocodeV1.h"
#include "hccl_sim_aicpu_stub.h"

#define SEM_NAME "/Hccl_Sim"

enum class ShmLockType { SHM_LOCK_SOCKET, SHM_LOCK_STREAM, SHM_LOCK_CCU, SHM_LOCK_INVALID };
enum class FakeSqeType { NOTIFY_WAIT, NOTIFY_RECORD, SDMA_REDUCE, MEM_CPY, CCU_SQE, HCCL_LABEL };
void SetFakeSocVersionStub(std::string socVersion);
const char *GetFakeSocVersionStub();

void H2DSignalHandle(int sig);
void H2DSignalHandle2(int sig);
void H2DSignalHandle3(int sig);

void ExitSubProcess(int rankId);
bool CreateDeviceProcesses(int ranksize);

#define MAX_IP_ADDR_NUM 448  // 框内最大ip地址数
#define MAX_DEVICE_NUM_PER_SERVER 32
#define MAX_STREAM_NUM (MAX_DEVICE_NUM_PER_SERVER * MAX_DEVICE_NUM_PER_SERVER)  // 最大流数，rank两两之间且带方向
#define MAX_NOTIFY_NUM 10000
#define DEVICE_SOCKET_SEND_MAX 65536  // device网卡通道限制64KB
#define SQE_SIZE 1024 * 2             // 200m  allgather  1524
constexpr int SHIFT_BIT32 = 32;
constexpr int PI_NUM_TWO = 2;

struct FakeD2HSqe {
    uint16_t streamId;
    FakeSqeType type;
    int notifyId;  
    int notifyCnt{1};
    void *dst;
    const void *src;
    uint64_t count;  // 代表数据实际长度，并不是 dataSize;  count  和length的关系
    rtDataType_t dataType;
    rtRecudeKind_t reduceOp;
};

struct ShareMemManager {
    int shm_id;
    int rank_size{0};
    void *shmPtr{nullptr};
    // 构造函数
    ShareMemManager(int shm_id, int rank_size, void *shmPtr) : shm_id(shm_id), rank_size(rank_size), shmPtr(shmPtr)
    {}
    // 析构函数
    ~ShareMemManager()
    {}
};

struct ShmCommon {
    int rankId;
    int shmid;
    int pid_h;
    int pid_d;
    int memPageId{0};
    void *basePtr;
    char signal_cmd;  // 0--init 1--lauch
    bool device_flag;
    int erro_flag;
};

struct ShmAicpu {
    KFCTaskCommStub kfcComm;
    HcclKernelParamLiteStub hcclKernelParam;
    u64 args;
    FakeD2HSqe d2hSqe[SQE_SIZE];
    int cnt{0};
};

struct ShmCcu {
    std::array<uint64_t, SimCcuV1::CCU_RESOURCE_XN_MAX>  xn[DIE_NUM];
    std::array<uint64_t, SimCcuV1::CCU_RESOURCE_GSA_MAX> gsa[DIE_NUM];
    std::array<uint16_t, SimCcuV1::CCU_RESOURCE_CKE_MAX> cke[DIE_NUM];
    std::array<char, SimCcuV1::CCU_RESOURCE_MS_SIZE>     ms[DIE_NUM];
};

struct ShmHead {
    ShmCommon com;
    ShmAicpu  aicpu;
    ShmCcu    ccu;
};

struct ShmMemory {
    char wholeMem[0];
};

struct ShmCb {
    ShmHead   head;
    ShmMemory memory;
};

struct SocketFd {
    int fd;
    char sendBuff[DEVICE_SOCKET_SEND_MAX];
    int totalLen;
};

struct ShmSocket {
    int socketHandleStore[MAX_DEVICE_NUM_PER_SERVER];
    SocketFd socketFds[MAX_STREAM_NUM];
    int jettyIdGen{0};
    u64 sqVaJettyIdMap[MAX_STREAM_NUM];
    int piValJettyIdMap[MAX_STREAM_NUM];
};

struct ShmStream {
    int notifyIdGen;
    int notifyIds[MAX_NOTIFY_NUM];
    int notifyCnts[MAX_NOTIFY_NUM];  // 记录每个notifyId可解锁的数量
    int streamIdGen;
    int streamIds[MAX_STREAM_NUM];
    bool verifyResults[MAX_DEVICE_NUM_PER_SERVER];
};

struct ShmPub {
    ShmSocket socket;
    ShmStream stream;
};

class ShmPoolLock {
public:
    ShmPoolLock()
    {
        sem_wait(sem_open(SEM_NAME, O_CREAT));
    };

    ~ShmPoolLock()
    {
        sem_post(sem_open(SEM_NAME, O_CREAT));
    };
};

class ShmPoolManager {
public:
    ShmPoolManager(bool isAicpu, int rankSize) : aicpuFlag_(isAicpu) {
        ranksize_ = rankSize;
        per_slice_size = 1024;                             // 每个分片1k(固定)
    };
    ~ShmPoolManager() = default;

public:
    bool CreateShmPool(Situation &situation);
    void DestroyShmPool();

    void SetDeviceId(int rankId);
    void SetSignalCmd(int rankId, char cmd, u64 args);
    u64 SetTilingData(int rankId, u64 contextAddr, u64 tilingData);
    u64 SetKernelParamLite(int rankId, void *args);

    void SetShmMemMgr(std::shared_ptr<ShareMemManager> &shmMgr);

    bool GetDeviceFlag(int rankId);
    void GetSqeTasks();
    void *GetShmCbBaseByRank(int rankId);
    
    // 用于910D中notifyId与notifyAddr的转换
    u64 GetNotifyAddr(int notifyId);
    int GetNotifyId(u64 notifyAddr);

    int GetShmId();
    ShmPub *GetShmPub();
    u64 GetMemSliceNum();
    u64 GetMemDeviceSize();
    void *GetShmBaseAddr();
    int CalcMem(void **devPtr, uint64_t size);
    bool CheckIfDeviceAddress(void *devPtr);
    
    void SetMemSliceNum(u64 num);
    void SetMemDeviceSize(u64 size);
    void SetVerifyResult(int rankId, bool result);
    bool GetVerifyResult(int rankId);

private:
    void *readMemoryMap(int pid);
    void CloseAndRemoveSemaphore();

    void LockShm(int rankId);
    void UnLockShm(int rankId);
    void InitialLock(int ranksize);

    key_t GenerateKey();
    void initialMgr();
    bool GenerateData(void *shm_ptr, int ranksize);

    void PreSetParamHost(Situation &situation);

    void DestroyShmData(); // device侧解除映射
    void DestroyShmDataEnd(); // host侧解除映射

private:
    int rankId_{0};
    int ranksize_{0};
    u64 shmSize_{0};
    u64 per_slice_size{0};
    u64 mem_slice_num{0};
    u64 mem_device_size{0};
    bool aicpuFlag_{false};
    ShmPub *shmPub_{nullptr};
    sem_t *sem{nullptr};
    std::shared_ptr<ShareMemManager> g_shareMemMgr;
};
#endif