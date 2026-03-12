/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: fake stream
 */

#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
#include "SimRunnerMgr.h"
#include "CcuResourceManager.h"

using namespace std;

void SimRunnerMgr::Init(int rankSize, bool isAicpu,  bool isCcu, CcuVersion ccuVersionFlag, bool caModelFlag, bool mc2Flag, std::shared_ptr<ShmPoolManager> &shmMgr)
{
    rankSize_ = rankSize;
    shmPoolMgr_ = shmMgr;
    aicpuFlag_ = isAicpu;
    ccuFlag_ = isCcu;
    ccuVersionFlag_ = ccuVersionFlag;
    caModelFlag_ = caModelFlag;
    mc2Flag_ = mc2Flag;
    CreateFakeStreamMgr();
    CreatFakeSocket();
    CreateNetDeviceMgr();
}

void SimRunnerMgr::Init(bool isAicpu, std::shared_ptr<ShmPoolManager> &shmMgr)
{
    shmPoolMgr_ = shmMgr;
    aicpuFlag_ = isAicpu;
    CreateFakeStreamMgr();
}

void SimRunnerMgr::Destory()
{
    DeleteFakeSocket();
    DeleteFakeStreamMgr();
    DeleteNetDeviceMgr();
    shmPoolMgr_->DestroyShmPool();  // 释放共享内存池
    aicpuFlag_ = false;
}

void SimRunnerMgr::CreatFakeSocket()
{
    if (fakeSocket_ != nullptr) {
        delete fakeSocket_;
        fakeSocket_ = nullptr;
    }
    fakeSocket_ = new FakeSocket(aicpuFlag_, shmPoolMgr_->GetShmPub());

    if (fakeUb_ != nullptr) {
        delete fakeUb_;
        fakeUb_ = nullptr;
    }
    fakeUb_ = new FakeUb();
}

void SimRunnerMgr::CreateNetDeviceMgr()
{
    if (netDeviceMgr_ != nullptr) {
        delete netDeviceMgr_;
        netDeviceMgr_ = nullptr;
    }
    netDeviceMgr_ = new NetDeviceMgr(rankSize_, ccuFlag_);
}

void SimRunnerMgr::CreateFakeStreamMgr()
{
    if (fakeStreamMgr_ != nullptr) {
        delete fakeStreamMgr_;     // 如果指针不为空，先释放原有的内存
        fakeStreamMgr_ = nullptr;  // 防止悬空指针
    }
    fakeStreamMgr_ = new FakeStreamMgr(rankSize_, aicpuFlag_, ccuFlag_, shmPoolMgr_->GetShmBaseAddr());
}

void SimRunnerMgr::DeleteFakeStreamMgr()
{
    if (fakeStreamMgr_ != nullptr) {
        delete fakeStreamMgr_;     // 如果指针不为空，先释放原有的内存
        fakeStreamMgr_ = nullptr;  // 防止悬空指针
    }
}

void SimRunnerMgr::DeleteFakeSocket()
{
    if (fakeSocket_ != nullptr) {
        delete fakeSocket_;
        fakeSocket_ = nullptr;
    }

    if (fakeUb_ != nullptr) {
        delete fakeUb_;
        fakeUb_ = nullptr;
    }
}

void SimRunnerMgr::DeleteNetDeviceMgr()
{
    if (netDeviceMgr_ != nullptr) {
        delete netDeviceMgr_;
        netDeviceMgr_ = nullptr;
    }
}

FakeSocket* SimRunnerMgr::GetFakeSocket()
{
    if (fakeSocket_ == nullptr) {
        std::cout<<"[SimRunnerMgr] GetFakeSocket is nullptr"<<endl;
    }
    return fakeSocket_;
}

FakeUb *SimRunnerMgr::GetFakeUb()
{
    if (fakeUb_ == nullptr) {
        std::cout << "[SimRunnerMgr] GetfakeUb is nullptr" << endl;
    }
    return fakeUb_;
}

FakeStreamMgr *SimRunnerMgr::GetFakeStreamMgr()
{
    return fakeStreamMgr_;
}

NetDeviceMgr *SimRunnerMgr::GetNetDeviceMgr()
{
    if (netDeviceMgr_ == nullptr) {
        std::cout << "[SimRunnerMgr] GetNetDeviceMgr is nullptr" << endl;
    }
    return netDeviceMgr_;
}

std::shared_ptr<ShmPoolManager> SimRunnerMgr::GetShmPoolMgr()
{
    return shmPoolMgr_;
}

void SimRunnerMgr::SetDeviceId(int rankId)
{
    rankId_ = rankId;
    if (ccuFlag_) {
        std::cout << "[SimRunnerMgr] ccuFlag_ is true, init ccu resource " << rankId_<< std::endl;
        CcuResouceManager::GetInstance().Init(rankId, rankSize_, CcuVersion::CCU_V1);
    }
    if (fakeStreamMgr_ != nullptr) {
        fakeStreamMgr_->SetDeviceId(rankId);
    }
    if (fakeSocket_ != nullptr) {
        fakeSocket_->SetDeviceId(rankId);
    }
    if (fakeUb_ != nullptr) {
        fakeUb_->SetDeviceId(rankId);
    }
    if (shmPoolMgr_.get() != nullptr) {
        shmPoolMgr_ ->SetDeviceId(rankId);
    }
    if (netDeviceMgr_ != nullptr) {
        netDeviceMgr_ ->SetDeviceId(rankId);
    }
}

int SimRunnerMgr::GetDeviceId()
{
    return rankId_;
}

pid_t SimRunnerMgr::GetPid(int rankId)
{
    return pid_d_[rankId];
}

void SimRunnerMgr::SetPid(int rankId, pid_t pid)
{
    pid_d_[rankId] = pid;
}

void SimRunnerMgr::ResizePid(uint32_t size, int value)
{
    pid_d_.resize(size, value);
}

bool SimRunnerMgr::IsAicpuSim()
{
    return aicpuFlag_;
}

void SimRunnerMgr::SetCcuFeatureFlag(bool ccuFlag)
{
    ccuFlag_ = ccuFlag;
}

void SimRunnerMgr::SetCaModelFlag(bool caModelFlag)
{
    caModelFlag_ = caModelFlag;
}

bool SimRunnerMgr::GetCcuFeatureFlag()
{
    return ccuFlag_;
}

bool SimRunnerMgr::GetMc2Flag()
{
    return mc2Flag_;
}

bool SimRunnerMgr::GetCaModelFlag()
{
    return caModelFlag_;
}

CcuVersion SimRunnerMgr::GetCcuVersionFlag()
{
    return ccuVersionFlag_;
}