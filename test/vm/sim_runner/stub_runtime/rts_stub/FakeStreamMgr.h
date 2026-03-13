/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: fake stream header
 */

#ifndef HCCL_SIM_FAKESTREAMMGR_H
#define HCCL_SIM_FAKESTREAMMGR_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include "hccl_sim_pub_stub.h"
#include "ccu_device_manager.h"
#include "ccu_microcode.h"
#include "CcuSimulator.h"

struct FakeSqe {
    FakeSqeType type;
    int notifyId;
    int notifyCnt{1};
    void *dst;
    const void *src;
    uint64_t count;  // 代表数据实际长度，并不是 dataSize;
    rtDataType_t dataType;
    rtRecudeKind_t reduceOp;
    // CCU节点信息
    rtCcuTaskInfo_t ccuTaskInfo;
    int devId;
    // FFTS子图节点信息
    int predCnt{0};
    std::vector<int> succList;
    bool isLastNode{false};  // 是否是FFTS+子图的最后一个节点
};

struct CcuInstrData {
    std::vector<Hccl::CcuRep::CcuInstr> instrData;
    uint16_t instrCnt{0};
};

class FakeNotifyMgr {
public:
    void Init(bool aicpuFlag, void *shmBase);
    int *CreateNotify(int rank);
    bool Record(const FakeSqe &sqe);
    bool Wait(FakeSqe &sqe);
    void DestroyNotify(int *notifyId);
    int GetRankIdByNotifyId(int notifyId);

private:
    bool aicpuFlag_{false};
    ShmPub *shmPub_{nullptr};
    std::mutex fakeNotifyMutex_;
    std::map<int, int> notifyCnts_;
    std::map<int, int> notifyRanks_;
};

class FakeStreamMgr {
public:
    FakeStreamMgr(int rankSize, bool isAicpu, bool ccuFlag, void *shmBase) : rankSize_(rankSize), aicpuFlag_(isAicpu), ccuFlag_(ccuFlag) {
        shmPub_ = reinterpret_cast<ShmPub *>(shmBase);
        shmCb_  = reinterpret_cast<char *>(shmBase) + sizeof(ShmPub);
        fakeNotifyMgr_.Init(isAicpu, shmBase);
    };
    void SetDeviceId(int rankId);
    int *CreateStream();
    void Sync(int streamId);
    void Append(int streamId, FakeSqe sqe);
    void SaveInstr(int ccuId, Hccl::CustomChannelInfoIn &instr);
    void AppendGraph(int streamId, int graphId, FakeSqe sqe);
    void DestroyStream(int *streamId);
    FakeNotifyMgr *GetFakeNotifyMgr();
    uint8_t *GetSqBufferAddr();
    int GetSqHead(uint32_t sqId);
    void UpdataSqHead(uint32_t sqId, int head);

private:
    bool ExecuteCcuSqe(FakeSqe &sqe);
    bool ExecuteSqe(FakeSqe &sqe, FakeNotifyMgr *notifyMgr);
    void PrintCcuTaskInfo();
    void GeneCaModelFile();
    bool HasSqe();
    bool HasGraph(int index);
    void StarsStreamSync();
    void FftsStreamSync();
    int FindGraphIndex(int streamId, int graphId);

private:
    bool aicpuFlag_{false};
    bool ccuFlag_{false};
    ShmPub *shmPub_{nullptr};
    char   *shmCb_{nullptr};
    int rankId_{0};
    int rankSize_{0};
    std::mutex fakeStreamMutex_;
    FakeNotifyMgr fakeNotifyMgr_;
    array<CcuInstrData, DIE_NUM> ccuInstrData_;  // 每个<devid, dieid>对应一个ccu instruction信息
    std::map<int, std::vector<FakeSqe>> sqeQueues_;                      // <streamId, sqe queue>
    std::vector<std::map<int, std::map<int, FakeSqe>>> graphGroups_;  // FFTS+模式存放图结构SQE
    std::array<std::shared_ptr<CcuSimulator>, DIE_NUM> ccuSimulators_;
    std::map<uint32_t, int> sqHeadMap_;  // 记录每条流在下发SQE时的头指针
    uint8_t sqBuffer_[hccl::HCCL_SQE_SIZE * hccl::HCCL_SQE_MAX_CNT];  // Device侧所有流共享同一个sqBuffer
};

#endif // HCCL_SIM_FAKESTREAMMGR_H
