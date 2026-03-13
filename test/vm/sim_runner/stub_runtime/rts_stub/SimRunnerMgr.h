/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: fake stream header
 */

#ifndef HCCL_SIM_RUNNERMGR_H
#define HCCL_SIM_RUNNERMGR_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "FakeStreamMgr.h"
#include "../hccp_stub/fake_socket.h"
#include "CcuResourceManager.h"

using namespace std;
class SimRunnerMgr {
public:
    SimRunnerMgr(const SimRunnerMgr&) = delete;
    SimRunnerMgr& operator=(const SimRunnerMgr&) = delete;

    static SimRunnerMgr& GetInstance() {
        static SimRunnerMgr instance;
        return instance;
    }
    // Host进程初始化
    void Init(int rankSize, bool isAicpu,  bool isCcu, CcuVersion ccuVersionFlag, bool caModelFlag, bool mc2Flag, std::shared_ptr<ShmPoolManager> &shmMgr);
    void Init(bool isAicpu, std::shared_ptr<ShmPoolManager> &shmMgr);  // Device进程初始化
    void Destory();

    // 多线程场景，使用线程安全的操作接口
    FakeUb* GetFakeUb();
    FakeSocket* GetFakeSocket();
    FakeStreamMgr *GetFakeStreamMgr();
    NetDeviceMgr *GetNetDeviceMgr();
    std::shared_ptr<ShmPoolManager> GetShmPoolMgr();

    void SetDeviceId(int rankId);
    int GetDeviceId();
    pid_t GetPid(int rankId);
    void SetPid(int rankId, pid_t pid);
    void ResizePid(uint32_t size, int value);

    bool IsAicpuSim();
    void SetCcuFeatureFlag(bool ccuFlag);
    void SetCaModelFlag(bool caModelFlag);
    bool GetCcuFeatureFlag();
    bool GetMc2Flag();
    CcuVersion GetCcuVersionFlag();
    bool GetCaModelFlag();

private:
    SimRunnerMgr() = default;
    ~SimRunnerMgr() = default;

    void CreatFakeSocket();
    void DeleteFakeSocket();
    void CreateFakeStreamMgr();
    void DeleteFakeStreamMgr();
    void CreateNetDeviceMgr();
    void DeleteNetDeviceMgr();

private:
    bool aicpuFlag_{false};
    int rankId_{0};
    int rankSize_{0};
    bool ccuFlag_{0};
    CcuVersion ccuVersionFlag_{CcuVersion::CCU_INVALID};
    bool mc2Flag_{0};
    bool caModelFlag_{0};
    vector<pid_t> pid_d_ {};
    FakeUb *fakeUb_{nullptr};
    FakeSocket *fakeSocket_{nullptr};
    FakeStreamMgr *fakeStreamMgr_{nullptr};
    NetDeviceMgr *netDeviceMgr_{nullptr};
    std::shared_ptr<ShmPoolManager> shmPoolMgr_;
};

#endif // HCCL_SIM_RUNNERMGR_H
