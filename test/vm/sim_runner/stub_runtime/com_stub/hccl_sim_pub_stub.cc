/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim public stub
 */

#include "hccl_sim_pub_stub.h"
#include "sqe_v82.h"
#include "sqe.h"
#include <fstream>
#ifdef DEVICE_STUB
#endif

#include "SimRunnerMgr.h"

#ifndef SIGUSR3
#define SIGUSR3 30  // 30是SIGUSR3的常见编号，具体值可以根据平台调整
#endif
std::string fakeSocVersion;

void H2DSignalHandle(int sig)
{  // 对方发过来一个就自动返回
    // 子进程 B 的信号处理函数
    if (sig == SIGUSR1) {
        HCCL_DEBUG("DEVICE: Received signal SIGUSR1, handling it in device program");
    }
}

void H2DSignalHandle2(int sig)
{  // 对方发过来一个就自动返回
    // 子进程 B 的信号处理函数
    if (sig == SIGUSR2) {
        std::cout << "DEVICE: Received signal SIGUSR2, exiting..." << std::endl;
        exit(0);
    }
}

void H2DSignalHandle3(int sig)
{  // 对方发过来一个就自动返回
    // 子进程 B 的信号处理函数
    if (sig == SIGUSR3) {
        std::cout << "DEVICE: Received signal SIGUSR3, exiting..." << std::endl;
        int32_t rankId = 0;
        aclrtGetDevice(&rankId);
        ShmCb *shmcb = reinterpret_cast<ShmCb *>(SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetShmCbBaseByRank(rankId));
        shmcb->head.com.erro_flag=-1;
    }
}

// ShmPoolManager BEGIN
bool ShmPoolManager::CreateShmPool(Situation &situation)
{
    DestroyShmPool();
    PreSetParamHost(situation);//全局参数准备
    key_t shm_key = GenerateKey();
    size_t shm_size = ((u64)sizeof(ShmCb)+ mem_device_size) * (u64)ranksize_ + (u64)sizeof(ShmPub);
    int shm_id = shmget(shm_key, shm_size, IPC_CREAT | 0666);
    if (shm_id == -1) {
        HCCL_ERROR("[CreateShmPool] START shmget failed");
        return false;
    }

    void *ptr = readMemoryMap(getpid());
    void *shm_ptr = shmat(shm_id, ptr, 0);
    memset(ptr, 0, shm_size);
    std::cout << "创建共享内存" << std::endl;

    if (shm_ptr == (void *)-1) {
        HCCL_ERROR("[CreateShmPool] START shmat failed");
        return false;
    }
    g_shareMemMgr = std::make_shared<ShareMemManager>(shm_id, ranksize_, shm_ptr);
    GenerateData(shm_ptr, ranksize_);
    return true;
}

void ShmPoolManager::DestroyShmPool()
{
    DestroyShmDataEnd();
    if (shmPub_ != nullptr) {
        shmPub_ = nullptr;
    }
    CloseAndRemoveSemaphore();
}

void ShmPoolManager::SetDeviceId(int rankId)
{
    rankId_ = rankId;
}

void ShmPoolManager::SetSignalCmd(int rankId, char cmd, u64 args)
{
    void *shm_ptr = GetShmCbBaseByRank(rankId);
    ShmCb *shmcb = reinterpret_cast<ShmCb *>(shm_ptr);
    shmcb->head.com.signal_cmd = cmd;
    shmcb->head.com.device_flag = true;
    shmcb->head.aicpu.args = args;
}

u64 ShmPoolManager::SetTilingData(int rankId, u64 contextAddr, u64 tilingData)
{
    void *shm_ptr = GetShmCbBaseByRank(rankId);
    ShmCb *shmcb = reinterpret_cast<ShmCb *>(shm_ptr);
    shmcb->head.aicpu.kfcComm.tilingData = tilingData;  // 把这个改一下就行,直接对共享内存操作
    shmcb->head.aicpu.kfcComm.context = contextAddr;
    u64 tempKfc = reinterpret_cast<u64>(&(shmcb->head.aicpu.kfcComm));  // 这块地址就是KFCTaskComm的地址
  
    return tempKfc;
}

u64 ShmPoolManager::SetKernelParamLite(int rankId, void *args)
{
    void *shm_ptr = GetShmCbBaseByRank(rankId);
    ShmCb *shmcb = reinterpret_cast<ShmCb *>(shm_ptr);

    auto *kernelParam = reinterpret_cast<HcclKernelLaunchParamStub *>(args);
    shmcb->head.aicpu.hcclKernelParam = kernelParam->kernel;
  
    u64 tempKernelParam = reinterpret_cast<u64>(&(shmcb->head.aicpu.hcclKernelParam));
    return tempKernelParam;

}

bool ShmPoolManager::GetDeviceFlag(int rankId)
{
    void *shm_ptr = GetShmCbBaseByRank(rankId);
    ShmCb *shmcb = reinterpret_cast<ShmCb *>(shm_ptr);
    return shmcb->head.com.device_flag;
}

void ShmPoolManager::GetSqeTasks()
{  // 脱离rankid  gethost怎么搞
    //  取共享内存里的所有sqe赛进去
    if (!aicpuFlag_) {
        return;
    }
    ShmCb *shmcb = reinterpret_cast<ShmCb *>(GetShmCbBaseByRank(rankId_));
    for (int j = 0; j < shmcb->head.aicpu.cnt; j++) {
        switch (shmcb->head.aicpu.d2hSqe[j].type) {
            case FakeSqeType::NOTIFY_RECORD:
                HCCL_DEBUG("GetSqe NOTIFY_RECORD, notifyId[%d], streamId[%d]",
                    (int)shmcb->head.aicpu.d2hSqe[j].streamId,
                    (int)shmcb->head.aicpu.d2hSqe[j].streamId);
                rtNotifyAicpuRecord(shmcb->head.aicpu.d2hSqe[j].notifyId, shmcb->head.aicpu.d2hSqe[j].notifyCnt, (int)shmcb->head.aicpu.d2hSqe[j].streamId);
                break;

            case FakeSqeType::NOTIFY_WAIT:
                HCCL_DEBUG(" GetSqe NOTIFY_WAIT, notifyId[%d], streamId[%d]",
                    shmcb->head.aicpu.d2hSqe[j].notifyId,
                    (int)shmcb->head.aicpu.d2hSqe[j].streamId);
                rtNotifyAicpuWait(shmcb->head.aicpu.d2hSqe[j].notifyId, shmcb->head.aicpu.d2hSqe[j].notifyCnt, (int)shmcb->head.aicpu.d2hSqe[j].streamId);
                break;

            case FakeSqeType::SDMA_REDUCE:
                HCCL_DEBUG(" GetSqe SDMA_REDUCE, dstAddr[%p], srcAddr[%p], streamId[%d]",
                    shmcb->head.aicpu.d2hSqe[j].dst,
                    shmcb->head.aicpu.d2hSqe[j].src,
                    (int)shmcb->head.aicpu.d2hSqe[j].streamId);
                rtReduceAicpuAsync(shmcb->head.aicpu.d2hSqe[j].dst,
                    (uint64_t)1024 * (uint64_t)1024 * (uint64_t)1024,
                    shmcb->head.aicpu.d2hSqe[j].src,
                    shmcb->head.aicpu.d2hSqe[j].count,
                    shmcb->head.aicpu.d2hSqe[j].reduceOp,
                    shmcb->head.aicpu.d2hSqe[j].dataType,
                    shmcb->head.aicpu.d2hSqe[j].streamId);
                break;

            case FakeSqeType::MEM_CPY:
                HCCL_DEBUG(" GetSqe MEM_CPY, dstAddr[%p], srcAddr[%p], streamId[%d]",
                    shmcb->head.aicpu.d2hSqe[j].dst,
                    shmcb->head.aicpu.d2hSqe[j].src,
                    (int)shmcb->head.aicpu.d2hSqe[j].streamId);
                rtMemcpyAicpuAsync(shmcb->head.aicpu.d2hSqe[j].dst,
                    (uint64_t)1024 * (uint64_t)1024 * (uint64_t)1024,
                    shmcb->head.aicpu.d2hSqe[j].src,
                    shmcb->head.aicpu.d2hSqe[j].count,
                    RT_MEMCPY_RESERVED,
                    shmcb->head.aicpu.d2hSqe[j].streamId);
                break;

            default:
                HCCL_ERROR("[GetSqeTasks] sqe type error");
                break;
        }
    }
}

void ShmPoolManager::SetShmMemMgr(std::shared_ptr<ShareMemManager> &shmMgr)
{
    g_shareMemMgr = shmMgr;
}

int ShmPoolManager::GetShmId()
{
    return g_shareMemMgr->shm_id;
}

void *ShmPoolManager::GetShmBaseAddr()
{
    if (g_shareMemMgr != nullptr) {
        return g_shareMemMgr->shmPtr;
    }
    return nullptr;
}

// memory related functions
int ShmPoolManager::CalcMem(void **devPtr, uint64_t size)
{
    int32_t rankId = 0;
    aclrtGetDevice(&rankId);
    ShmCb *devRankBase = reinterpret_cast<ShmCb *>(GetShmCbBaseByRank(rankId));

    int sliceCnt = (size + per_slice_size - 1) / per_slice_size;  // 向上取整
    if (devRankBase->head.com.memPageId + sliceCnt >= mem_slice_num) {
        HCCL_ERROR("[CalcMem]: Malloc memory for rank[%d], failed. size=[%lu], slice size[%u][%u][%u]", rankId, size, mem_slice_num, sliceCnt, devRankBase->head.com.memPageId);
        return -1;  // 内存不足，分配失败
    }
    uint64_t devAddr =
        reinterpret_cast<uint64_t>(devRankBase->memory.wholeMem) + per_slice_size * devRankBase->head.com.memPageId;
    *devPtr = reinterpret_cast<void *>(devAddr);
    devRankBase->head.com.memPageId += sliceCnt;
    // host侧无法初始化dev映射地址，采用host地址代替（指向的是同一块内存）
    memset(*devPtr, 0, sliceCnt * per_slice_size);
    return 0;
}

u64 ShmPoolManager::GetNotifyAddr(int notifyId)
{
    if (notifyId >= MAX_NOTIFY_NUM) {
        return -1;
    }

    ShmPub *shmPub = GetShmPub();
    if (shmPub == nullptr) {
        return -1;
    }

    return reinterpret_cast<u64>(&(shmPub->stream.notifyIds[notifyId]));
}

int ShmPoolManager::GetNotifyId(u64 notifyAddr)
{
    ShmPub *shmPub = GetShmPub();
    if (shmPub == nullptr) {
        return -1;
    }

    for (int i = 0; i < MAX_NOTIFY_NUM; i++) {
        if (reinterpret_cast<u64>(&(shmPub->stream.notifyIds[i])) == notifyAddr) {
            return i;
        }
    }

    return -1;
}

// 获取指定rankid的shm块基址，ishost标记区分是host侧调用还是device侧调用，返回值要据此map到相应的虚拟地址
void *ShmPoolManager::GetShmCbBaseByRank(int rankId)
{
    // 基址只通过host/8个device侧区别，哪个进程想掉用哪个
    void *shm_ptr = (char *)GetShmBaseAddr() + sizeof(ShmPub) + (sizeof(ShmCb)+ mem_device_size) * rankId;
    if (shm_ptr == (void *)-1) {
        std::cout << "shmat failed" << std::endl;
    }

    return shm_ptr;
}

ShmPub *ShmPoolManager::GetShmPub()
{
    if (!aicpuFlag_) {
        return shmPub_;
    } else {
        void *shm_ptr = GetShmBaseAddr();  // 调用他的都是主进程？
        return reinterpret_cast<ShmPub *>(shm_ptr);
    }
}

bool ShmPoolManager::CheckIfDeviceAddress(void *devPtr)
{
    // 通过判断地址是否在host侧共享内存映射地址范围来判断是否需要转换
    char *devPtrIn = reinterpret_cast<char *>(devPtr);
    char *devMemBase = (char *)GetShmBaseAddr();
    char *devkMaxAddr = devMemBase + shmSize_;

    if (devPtrIn < devMemBase || devPtrIn > devkMaxAddr) {
        return false;
    }
    return true;
}
u64 ShmPoolManager::GetMemSliceNum()
{
    return this->mem_slice_num;
}

u64 ShmPoolManager::GetMemDeviceSize()
{
    return this->mem_device_size;
}

void ShmPoolManager::SetMemSliceNum(u64 num)
{
    this->mem_slice_num = num;
}

void ShmPoolManager::SetMemDeviceSize(u64 size)
{
    this->mem_device_size = size;
}

void ShmPoolManager::SetVerifyResult(int rankId, bool result)
{
    std::cout << "设置VerifyResult rank:" << rankId << " result:" << result << std::endl;
    shmPub_->stream.verifyResults[rankId] = result;
}

bool ShmPoolManager::GetVerifyResult(int rankId)
{
    return shmPub_->stream.verifyResults[rankId];
}

void ShmPoolManager::InitialLock(int ranksize)
{
    sem_open(SEM_NAME, O_CREAT | O_EXCL, 0666, 1);
    for (int i = 0; i < ranksize; i++) {
        std::string semName = std::to_string(i);
        sem_open(semName.c_str(), O_CREAT | O_EXCL, 0666, 1);  // 创建信号量
    }
}

key_t ShmPoolManager::GenerateKey()
{
    // 使用string生成哈希
    std::hash<std::string> hash_fn;
    size_t hash = hash_fn("hccl_sim");
    if (hash < 0) {
        hash = -hash;
    }
    // 将哈希值转换为key_t（通常是32位的）
    return static_cast<key_t>(hash);
}

void *ShmPoolManager::readMemoryMap(int pid)
{
    std::string filename = "/proc/" + std::to_string(pid) + "/maps";
    std::ifstream mapsFile(filename);

    if (!mapsFile.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 0;
    }

    u64 totalStart = std::numeric_limits<u64>::max();
    u64 totalEnd = 0;
    std::vector<std::pair<u64, u64>> ranges;

    std::string line;
    while (std::getline(mapsFile, line)) {
        if (line.empty() || line[0] == '[') continue;

        std::stringstream ss(line);
        std::string addressRange;
        ss >> addressRange;

        size_t hyphenPos = addressRange.find('-');
        if (hyphenPos != std::string::npos) {
            u64 startAddr, endAddr;
            std::string prefix = addressRange.substr(0, hyphenPos);
            std::string suffix = addressRange.substr(hyphenPos + 1);

            std::stringstream startStream(prefix), endStream(suffix);
            startStream >> std::hex >> startAddr;
            endStream >> std::hex >> endAddr;

            totalStart = std::min(totalStart, startAddr);
            totalEnd = std::max(totalEnd, endAddr);
            ranges.emplace_back(startAddr, endAddr);
        }
    }

    std::sort(ranges.begin(), ranges.end());
    u64 longestStart = 0, longestLen = 0;
    u64 prevEnd = totalStart;

    for (const auto& range : ranges) {
        u64 start = range.first;
        u64 end = range.second;

        if (start > prevEnd) {
            u64 freeStart = prevEnd;
            u64 freeLen = start - freeStart;
            
            // 检查地址最高四位是否小于0x6
            if ((freeStart & 0xf00000000000) < 0x600000000000) {
                // std::cout << std::hex << "valid free region: 0x" << freeStart 
                //           << "-0x" << start << " (Length: " << std::dec << freeLen << ")\n";
                
                if (freeLen > longestLen) {
                    longestStart = freeStart;
                    longestLen = freeLen;
                }
            }
        }
        prevEnd = std::max(prevEnd, end);
    }

    if (longestLen > 0) {
        // std::cout << "\n最长有效空闲区域:\n"
        //           << std::hex << "起始地址: 0x" << longestStart << "\n长度: " << std::dec << longestLen << " 字节\n"
        //           << std::endl;
    } else {
        HCCL_ERROR("no valid free region found");
        return nullptr;
    }
    return reinterpret_cast<void *>(longestStart);
}

void ShmPoolManager::CloseAndRemoveSemaphore()
{
    sem = sem_open(SEM_NAME, O_CREAT);
    sem_close(sem);        // 关闭信号量
    sem_unlink(SEM_NAME);  // 删除信号量
    for (int i = 0; i < ranksize_; i++) {
        std::string semName = std::to_string(i);
        sem_close(sem_open(semName.c_str(), O_CREAT));  // 关闭信号量
        sem_unlink(semName.c_str());              // 删除信号量
    }
}

void ShmPoolManager::LockShm(int rankId)  // rank级别的共享内存加锁，用sev实现，以rankid作为sevname
{
    if (!aicpuFlag_) {
        return;
    }
    // 根据id生成信号量名字
    std::string semName = std::to_string(rankId);
    sem_wait(sem_open(semName.c_str(), 0));  // 请求锁
}

void ShmPoolManager::UnLockShm(int rankId)  // 释放信号量，解锁
{
    if (!aicpuFlag_) {
        return;
    }
    // 根据id生成信号量名字
    std::string semName = std::to_string(rankId);
    sem_post(sem_open(semName.c_str(), 0));  // 请求锁
}

void ShmPoolManager::initialMgr()
{
    shmPub_->socket.jettyIdGen = 0;
    shmPub_->stream.notifyIdGen = 0;
    shmPub_->stream.streamIdGen = 0;
    for (int i = 0; i < MAX_DEVICE_NUM_PER_SERVER; i++) {
        shmPub_->socket.socketHandleStore[i] = i;
        shmPub_->stream.verifyResults[i] = false;
    }
    for (int i = 0; i < MAX_STREAM_NUM; i++) {
        shmPub_->socket.socketFds[i].fd = -1;
        shmPub_->socket.sqVaJettyIdMap[i] = -1;
        shmPub_->socket.piValJettyIdMap[i] = 0;
    }
    for (int i = 0; i < MAX_NOTIFY_NUM; i++) {
        shmPub_->stream.notifyCnts[i] = 0;
    }
}

bool ShmPoolManager::GenerateData(void *shm_ptr, int ranksize)
{
    shmPub_ = reinterpret_cast<ShmPub *>(shm_ptr);
    initialMgr();

    for (int i = 0; i < ranksize; i++) {
        ShmCb *shmcb = reinterpret_cast<ShmCb *>((char *)shm_ptr + sizeof(ShmPub) + (sizeof(ShmCb) + mem_device_size) * i);
        shmcb->head.com.memPageId = 0;
        shmcb->head.com.rankId = i;
        shmcb->head.com.pid_h = getpid();
        shmcb->head.com.device_flag = false;  // 显示置为false
        for (int j = 0; j < SQE_SIZE; j++) {
            shmcb->head.aicpu.d2hSqe[j].notifyCnt = 1;
        }
    }

    return true;
}

void ShmPoolManager::PreSetParamHost(Situation &situation)
{
    InitialLock(ranksize_);//锁资源
    SimRunnerMgr::GetInstance().ResizePid(ranksize_, 0);
    if (situation.GetShmSize() == 0) {
        if (situation.GetOpType() == OpType::ALLGATHER) {
            mem_slice_num = 2.5 * 1024 * 1024;
        } else {
            mem_slice_num = 1 * 1024 * 1024;                   // 总共分配片数(默认)
        }
    } else {
        mem_slice_num = situation.GetShmSize();
    }
    mem_device_size = mem_slice_num * per_slice_size;
    shmSize_ = (sizeof(ShmCb)+ mem_device_size) * ranksize_ + sizeof(ShmPub);
}

// device侧解除映射
void ShmPoolManager::DestroyShmData()  // host侧，用例结束时，释放共享内存资源
{
    key_t shm_key = GenerateKey();
    int shm_id = shmget(shm_key, shmSize_, 0666);
    if (shm_id == -1) {
        std::cout << "DestroyShmData shmget failed" << std::endl;
        return ;
    }

    void *shm_ptr = shmat(shm_id, nullptr, 0);
    if (shm_ptr == (void *)-1) {
        std::cout << "shmat failed" << std::endl;
    }

    if (shmdt(shm_ptr) == -1) {
        std::cout << "shmdt failed in A process" << std::endl;
    }

    // 删除共享内存
    if (shmctl(shm_id, IPC_RMID, nullptr) == -1) {
        std::cout << "shmctl failed" << std::endl;
    }
}

void ShmPoolManager::DestroyShmDataEnd()  // host侧，用例结束时，释放共享内存资源
{
    std::cout << "销毁共享内存" << std::endl;
    if(g_shareMemMgr.get() == nullptr){
        std::cout<<"g_shareMemMgr->shmPtr is nullptr"<<std::endl;
        return;
    }
    
    if (shmdt(g_shareMemMgr->shmPtr) == -1) {
        std::cout << "shmdt failed in A process" << std::endl;
    }

    // 删除共享内存段
    if (shmctl(g_shareMemMgr->shm_id, IPC_RMID, nullptr) == -1) {
        std::cerr << "shmctl IPC_RMID failed\n";
        return;
    }
}
// ShmPoolManager END

void SetFakeSocVersionStub(std::string socVersion)
{
    fakeSocVersion = socVersion;
}

const char *GetFakeSocVersionStub()
{
    return fakeSocVersion.c_str();
}

void ExitSubProcess(int rankId)
{
    auto pid = SimRunnerMgr::GetInstance().GetPid(rankId);
    kill(pid, SIGUSR2);  // 子进程的pid
    int status;
    HCCL_DEBUG("[ExitSubProcess] waitpid start...");
    waitpid(pid, &status, 0);
}

const char *GetAICPUBinPath()
{
    const char *path_daily = "./hccl_sim_aicpu";
    const char *path_ci = "./output/llt/hccl_lib/hccl_sim_aicpu";
    if (access(path_daily, F_OK) == 0) {
        return path_daily;
    }

    return path_ci;
}

// 主进程是不是永远不用pause
bool CreateDeviceProcesses(int ranksize)
{
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    if (!simRunnerMgr.IsAicpuSim()) {
        return true;
    }
    signal(SIGUSR3, H2DSignalHandle3);
    auto shmPoolMgr = simRunnerMgr.GetShmPoolMgr();
    const char *fake_device_type = GetFakeSocVersionStub();
    for (int rankId = 0; rankId < ranksize; rankId++) {
        auto pid = fork();
        simRunnerMgr.SetPid(rankId, pid);
        if (pid < 0) {  // 不能用这个得用进程替换
            std::cerr << "Fork failed!" << std::endl;
            return false;
        }

        const char *aicpu_path = GetAICPUBinPath();
        if (pid == 0) {  // 子进程
            execlp(aicpu_path,
                "hccl_sim_aicpu",
                std::to_string(rankId).c_str(),
                std::to_string(shmPoolMgr->GetShmId()).c_str(),
                std::to_string(ranksize).c_str(),
                fake_device_type,
                std::to_string(reinterpret_cast<u64>(shmPoolMgr->GetShmBaseAddr())).c_str(),
                std::to_string(shmPoolMgr->GetMemSliceNum()).c_str(),
                std::to_string(shmPoolMgr->GetMemDeviceSize()).c_str(),
                nullptr);
        } else {       // 父进程
            sleep(1);  // 确保子进程 B 已经设置好信号处理函数并处于等待状态
            ShmCb *shmcb = reinterpret_cast<ShmCb *>(shmPoolMgr->GetShmCbBaseByRank(rankId));
            if (shmcb->head.com.erro_flag == -1) {
                HCCL_ERROR("[CreateDeviceProcesses] error");
                return false;
            }
        }
    }
    return true;
}