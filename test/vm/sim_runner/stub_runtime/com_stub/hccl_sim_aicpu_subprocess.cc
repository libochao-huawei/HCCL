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

u64 g_memSliceNum;
u64 g_memDeviceSize;

extern void InitHcclMockStub();
extern void log_level_set_stub(s32 log_level);

int deviceProcess(ShmCb *shmcb, void *shm_ptr)
{
#ifdef DEVICE_STUB
    u64 argsInfo = shmcb->head.aicpu.args;
    void *args = reinterpret_cast<void *>(argsInfo);
    if (shmcb->head.com.signal_cmd == '0') {
        uint32_t ret = RunAicpuKfcResInitV2(args);
    }
    if (shmcb->head.com.signal_cmd == '1') {
        uint32_t ret = RunAicpuRpcSrvLaunchV2(args);
        return -1;
    }
    if (shmcb->head.com.signal_cmd == '2') {
        uint32_t ret = HcclKernelEntrance(args);
        return -1;
    }
#endif
    return 0;
}

void PreSetParamDevice(char **argv)
{
    InitHcclMockStub();
    SetFakeSocVersionStub(std::string(argv[4]));
    g_memSliceNum = strtoull(argv[6], nullptr, 0);
    g_memDeviceSize = strtoull(argv[7], nullptr, 0);
    if (SalGetEnv("ASCEND_GLOBAL_LOG_LEVEL") != "EmptyString") {
        log_level_set_stub(std::stoi(SalGetEnv("ASCEND_GLOBAL_LOG_LEVEL")));
    }
}

int AicpuMain(int argc, char **argv)
{
    std::cout << "Device侧子进程创建成功" << std::endl;
    // 先映射地址    然后再存起来
    signal(SIGUSR1, H2DSignalHandle);
    signal(SIGUSR2, H2DSignalHandle2);
    int rankid = std::stoi(argv[1]);
    int shm_id = std::stoi(argv[2]);
    int rankSize = std::stoi(argv[3]);
    PreSetParamDevice(argv);//device侧全局参数设置

    char *endptr;
    uint64_t hostPtr = std::strtoull(argv[5], &endptr, 10);
    void *shm_ptr = shmat(shm_id, (const void *)hostPtr, 0);
    if (shm_ptr == (void *)-1) {
        HCCL_ERROR("shmat failed ...  Error code: %d, %s", errno, strerror(errno));
        kill(getppid(), SIGUSR3);
        return -1;
    }

    // 创建Device侧共享内存对象，初始化共享内存池参数(从Host侧传递来的)
    auto shmPoolMgr = std::make_shared<ShmPoolManager>(true, rankSize);
    shmPoolMgr->SetMemSliceNum(g_memSliceNum);
    shmPoolMgr->SetMemDeviceSize(g_memDeviceSize);
    // 初始化SimRunnerMgr实例
    SimRunnerMgr::GetInstance().Init(true, shmPoolMgr);
    aclrtSetDevice(rankid);
    auto shmMemMgr = std::make_shared<ShareMemManager>(shm_id, rankSize, shm_ptr); // shmptr放的是基址
    SimRunnerMgr::GetInstance().GetShmPoolMgr()->SetShmMemMgr(shmMemMgr);
    ShmCb *shmcb = reinterpret_cast<ShmCb *>(SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetShmCbBaseByRank(rankid));
    shmcb->head.com.pid_d = getpid(); // 通过pid  来kill
    shmcb->head.com.basePtr = shm_ptr;

    while (true) {
        if (shmcb->head.com.device_flag == true) {  // 如果device侧已经准备好了,就可以执行了
            int ret = deviceProcess(shmcb, shm_ptr);
            shmcb->head.com.device_flag = false;
            if (ret == -1) {
                if (shmdt(shm_ptr) == -1) {
                    HCCL_ERROR("SHMDT FAILED IN DEVICE PROCESS RANKID= %d", rankid);
                }
                return 0;
            }
        }
        if (shmcb->head.com.device_flag == false) {
            // std::cout << "Device侧子进程开始sleep" << std::endl;
            pause();
        }
    }
    return 0;
}