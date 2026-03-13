/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: fake stream
 */

#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
#include "FakeStreamMgr.h"
#include "log.h"
#include "fwk_types.h"
#include "CcuResourceManager.h"
#include "SimRunnerMgr.h"

using namespace std;

int g_timeout = 0;
time_t g_syncTime = time(nullptr);

bool IsStreamSyncTimeOut()
{
    time_t curTime = time(nullptr);
    if (curTime - g_syncTime > g_timeout) {
        HCCL_ERROR(
            "[FakeStreamMgr][Sync]time out[%d]s, sync_time[%d], current_time[%d]", g_timeout, g_syncTime, curTime);
        return true;
    }

    return false;
}

bool MemCpy(FakeSqe &sqe)
{
    memcpy(sqe.dst, sqe.src, sqe.count);
    return true;
}

// 对不同类型的做Sum的Reduce操作
template <typename T>
bool MemReduceSumType(void *dst_addr, const void *src_addr, uint64_t count)
{
    T *dst = (T *)(dst_addr);
    T *src = (T *)(src_addr);
    auto dataCount = count / sizeof(T);
    for (auto index = 0; index < dataCount; ++index) {
        dst[index] += src[index];
    }
    return true;
}

// 对不同类型的做MIN的Reduce操作
template <typename T>
bool MemReduceMinType(void *dst_addr, const void *src_addr, uint64_t count)
{
    T *dst = (T *)(dst_addr);
    T *src = (T *)(src_addr);
    auto dataCount = count / sizeof(T);
    for (auto index = 1; index < dataCount; ++index) {
        if (src[index] < dst[index]) {
            dst[index] = src[index];
        }
    }
    return true;
}

// 对不同类型的做MAX的Reduce操作
template <typename T>
bool MemReduceMaxType(void *dst_addr, const void *src_addr, uint64_t count)
{
    T *dst = (T *)(dst_addr);
    T *src = (T *)(src_addr);
    auto dataCount = count / sizeof(T);
    for (auto index = 1; index < dataCount; ++index) {
        if (src[index] > dst[index]) {
            dst[index] = src[index];
        }
    }
    return true;
}

bool MemReduce(FakeSqe &sqe)
{
    if (sqe.reduceOp == rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_ADD) {
        if (rtDataType_t::RT_DATA_TYPE_FP32 == sqe.dataType) {
            return MemReduceSumType<float>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT8 == sqe.dataType) {
            return MemReduceSumType<s8>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT16 == sqe.dataType) {
            return MemReduceSumType<s16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT32 == sqe.dataType) {
            return MemReduceSumType<s32>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT8 == sqe.dataType) {
            return MemReduceSumType<u8>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT16 == sqe.dataType) {
            return MemReduceSumType<u16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT32 == sqe.dataType) {
            return MemReduceSumType<u32>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_FP16 == sqe.dataType) {
            return MemReduceSumType<FP16>(sqe.dst, sqe.src, sqe.count);
        }
        // 继续添加其他类型
    } else if (sqe.reduceOp == rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MIN) {
        if (rtDataType_t::RT_DATA_TYPE_FP32 == sqe.dataType) {
            return MemReduceMinType<float>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT32 == sqe.dataType) {
            return MemReduceMinType<s32>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_FP16 == sqe.dataType) {
            return MemReduceMinType<FP16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT8 == sqe.dataType) {
            return MemReduceMinType<s8>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT16 == sqe.dataType) {
            return MemReduceMinType<s16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT8 == sqe.dataType) {
            return MemReduceMinType<u8>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT16 == sqe.dataType) {
            return MemReduceMinType<u16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT32 == sqe.dataType) {
            return MemReduceMinType<u32>(sqe.dst, sqe.src, sqe.count);
        }
        // 继续添加其他类型数据
    } else if (sqe.reduceOp == rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MAX) {
        if (rtDataType_t::RT_DATA_TYPE_FP32 == sqe.dataType) {
            return MemReduceMaxType<float>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT32 == sqe.dataType) {
            return MemReduceMaxType<s32>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_FP16 == sqe.dataType) {
            return MemReduceMaxType<FP16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT8 == sqe.dataType) {
            return MemReduceMaxType<s8>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_INT16 == sqe.dataType) {
            return MemReduceMaxType<s16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT8 == sqe.dataType) {
            return MemReduceMaxType<u8>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT16 == sqe.dataType) {
            return MemReduceMaxType<u16>(sqe.dst, sqe.src, sqe.count);
        } else if (rtDataType_t::RT_DATA_TYPE_UINT32 == sqe.dataType) {
            return MemReduceMaxType<u32>(sqe.dst, sqe.src, sqe.count);
        }
        // 继续添加其他类型数据
    }
    return false;
}

bool FakeStreamMgr::ExecuteCcuSqe(FakeSqe &sqe)
{
    int dieId = static_cast<int>(sqe.ccuTaskInfo.dieId);
    uint16_t instStartId = sqe.ccuTaskInfo.instStartId;
    uint16_t instCnt     = sqe.ccuTaskInfo.instCnt;
    uint32_t argSize     = sqe.ccuTaskInfo.argSize;
    HCCL_DEBUG("[ExecuteCcuSqe]: rankId=[%d], dieId=[%d], startInstrId=[%u], instrCnt=[%u], argSize=[%u]",
        sqe.devId, dieId, instStartId, instCnt, argSize);

    if (ccuSimulators_[dieId].get() != nullptr) {
        HCCL_DEBUG("[ExecuteCcuSqe] exist simulator....");
        CcuResouceManager::GetInstance().AddTaskInfo(dieId, sqe.ccuTaskInfo);
        for (uint32_t i = 0; i < argSize; i++) {
            HCCL_DEBUG("[FakeStreamMgr][ExecuteCcuSqe] argvalue, dieId=[%u], index=[%u], value=[%lu].", dieId, i, sqe.ccuTaskInfo.args[i]);
        }
        uint16_t endInstrIdTmp = instStartId + instCnt;
        ccuSimulators_[dieId]->Init(instStartId, endInstrIdTmp, instCnt);
        return ccuSimulators_[dieId]->Execute();
    }

    CcuResouceManager::GetInstance().AddTaskInfo(dieId, sqe.ccuTaskInfo);
    for (uint32_t i = 0; i < argSize; i++) {
        HCCL_DEBUG("[FakeStreamMgr][ExecuteCcuSqe] argvalue, dieId=[%u], index=[%u], value=[%lu].", dieId, i, sqe.ccuTaskInfo.args[i]);
    }

    uint16_t endInstrId = instStartId + instCnt;
    ccuSimulators_[dieId] = std::make_shared<CcuSimulator>(sqe.devId, dieId, instStartId, endInstrId, instCnt);
    return ccuSimulators_[dieId]->Execute();
}

bool FakeStreamMgr::ExecuteSqe(FakeSqe &sqe, FakeNotifyMgr *notifyMgr)
{
    if (sqe.type == FakeSqeType::CCU_SQE) {
        return ExecuteCcuSqe(sqe);
    }

    if (sqe.type == FakeSqeType::NOTIFY_RECORD) {
        return notifyMgr->Record(sqe);
    }

    if (sqe.type == FakeSqeType::NOTIFY_WAIT) {
        return notifyMgr->Wait(sqe);
    }

    if (sqe.type == FakeSqeType::MEM_CPY) {
        return MemCpy(sqe);
    }

    if (sqe.type == FakeSqeType::SDMA_REDUCE) {
        return MemReduce(sqe);
    }

    if (sqe.type == FakeSqeType::HCCL_LABEL) {
        // LABEL对应的sqe执行时不做任何处理，但需要执行成功并根据successorList解锁后面的sqe执行
        return true;
    }

    return false;
}

void FakeStreamMgr::SetDeviceId(int rankId)
{
    rankId_ = rankId;
}

int *FakeStreamMgr::CreateStream()
{
    ShmPoolLock shmPoolLock;

    if ((shmPub_ == nullptr) || (shmPub_->stream.streamIdGen >= MAX_STREAM_NUM)) {
        std::cout << "FakeStreamMgr::CreateStream invalid streamIdGen!" << std::endl;
        return nullptr;
    }

    int curStreamId = shmPub_->stream.streamIdGen++;
    shmPub_->stream.streamIds[curStreamId] = curStreamId;
    return &shmPub_->stream.streamIds[curStreamId];
}

bool FakeStreamMgr::HasSqe()
{
    int sum = 0;
    for (auto& iter : sqeQueues_) {
        sum += iter.second.size();
    }
    return sum > 0;
}

bool FakeStreamMgr::HasGraph(int index)
{
    int sum = 0;
    for (auto iter : graphGroups_[index]) {
        sum += iter.second.size();
    }
    return sum > 0;
}

void FakeStreamMgr::Sync(int streamId)
{
    std::cout << "开始执行流同步, pid:" << getpid() << std::endl;
    lock_guard<mutex> lock(fakeStreamMutex_);
    // 读取环境遍历时间判断SQE执行超时
    string timeOutEnv = SalGetEnv("HCCL_EXEC_TIMEOUT");
    g_timeout = stoi(timeOutEnv.empty() ? "600" : timeOutEnv);
    // g_timeout = 30; // 先设置为超时时间为30s
    g_syncTime = time(nullptr);
    
    if (HasSqe()) {
        // STARS模式下流内SQE串行执行
        std::cout << "STARS模式下流内SQE串行执行" << std::endl;
        StarsStreamSync();
    } else {
        // FFTS+模式下流内SQE图结构并行执行
        std::cout << "FFTS+模式下流内SQE图结构并行执行" << std::endl;
        FftsStreamSync();
    }
}

void FakeStreamMgr::PrintCcuTaskInfo()
{
    if (!ccuFlag_) {
        return;
    }
    // 打印sqe信息
    HCCL_DEBUG("==========================Dump ccu task info start===========================");
    for (auto &stream : sqeQueues_) {
        int streamIndex = 0;
        for (auto &sqe : stream.second) {
            auto taskInfo = sqe.ccuTaskInfo;
            HCCL_DEBUG("streamId=[%u], devId=[%d], missionId=[%d], instStartId=[%u], instCnt=[%u], key=[%lu], argSize=[%u].",
                stream.first, static_cast<int>(sqe.devId), static_cast<int>(taskInfo.missionId), taskInfo.instStartId, taskInfo.instCnt,
                taskInfo.key, taskInfo.argSize);
            for (u32 i = 0; i < taskInfo.argSize; i++) {
                HCCL_DEBUG("task arg index=[%u], value=[%lu].", i, taskInfo.args[i]);
            }
        }
    }
    HCCL_DEBUG("==========================Dump ccu task info end===========================");
}

void FakeStreamMgr::StarsStreamSync()
{
    PrintCcuTaskInfo();
    if (ccuFlag_) {
        CcuResouceManager::GetInstance().InitInstrInfo(ccuInstrData_, sqeQueues_);
    }
    for (auto &stream : sqeQueues_) {
        HCCL_INFO("开始执行streamId=[%d]的sqe, size=[%d], rankId=[%d]", stream.first, stream.second.size(), rankId_);
        for (auto &sqe : stream.second) {
            if (sqe.type == FakeSqeType::NOTIFY_WAIT || sqe.type == FakeSqeType::NOTIFY_RECORD) {
                HCCL_INFO("sqe type=[%d], notifyId=[%d], nofityCnt=[%d]", static_cast<int>(sqe.type), sqe.notifyId, sqe.notifyCnt);
            }
        }
    }
    // ccu V121的指令执行还没适配，暂时先返回
    if (SimRunnerMgr::GetInstance().GetCcuVersionFlag() == CcuVersion::CCU_V2) {
        return;
    }
    while (HasSqe()) {
        for (auto &stream : sqeQueues_) {
            auto id = stream.first;
            // 开始按序执行sqe，删掉执行完的sqe，碰到执行不下去的时候退出本次循环
            auto it = stream.second.begin();
            while (it != stream.second.end()) {
                if (IsStreamSyncTimeOut()) {
                    return;  // 判断超时退出
                }

                auto ret = ExecuteSqe(*it, &fakeNotifyMgr_);
                if (ret) {
                    it = stream.second.erase(it);
                    continue;
                }

                if (it->type == FakeSqeType::CCU_SQE) {
                    break;
                }

                if (it->type == FakeSqeType::NOTIFY_WAIT) {
                    break;
                }

                // 非NOTIFY_WAIT执行失败，直接退出同步
                HCCL_ERROR("ExecuteSqe failed, type=%d", static_cast<int>(it->type));
                return;
            }
        }
    }
}

void SearchGraphRoots(const map<int, FakeSqe> &graph, queue<int> &rootQue)
{
    // 查找当前子图的根节点(predCnt <= 0)
    for_each(graph.begin(), graph.end(), [&rootQue](pair<int, FakeSqe> a) {
        if (a.second.predCnt <= 0) {
            rootQue.push(a.first);
        }
    });
}

void UpdateSubGraghs(map<int, FakeSqe> &graph, int root, const vector<int> &succList, queue<int> &rootQue)
{
    for (auto &node : succList) {
        auto it = graph.find(node);
        if (it == graph.end()) {
            continue;
        }

        // 当前节点执行成功，更新succList节点的predCnt
        graph[node].predCnt--;
        if (graph[node].predCnt <= 0) {
            rootQue.push(node);  // 满足根节点条件的节点入队
        }
    }

    graph.erase(root);
}

void FakeStreamMgr::FftsStreamSync()
{
    for (int i = 0; i < graphGroups_.size(); i++) {
        // 每次遍历处理一组图结构(如先处理200MB)
        while (HasGraph(i)) {
            // 遍历执行每张卡的FFTS+子图结构
            for (auto &stream : graphGroups_[i]) {
                map<int, FakeSqe> &subGraph = stream.second;
                queue<int> rootQue;
                SearchGraphRoots(subGraph, rootQue);  // 查找子图根节点集合

                int failCnt = 0;
                while (!rootQue.empty()) {
                    int count = rootQue.size();
                    for (int i = 0; i < count; i++) {
                        if (IsStreamSyncTimeOut()) {
                            return;  // 判断超时退出
                        }

                        int root = rootQue.front();
                        rootQue.pop();
                        FakeSqe &sqe = subGraph[root];
                        bool ret = ExecuteSqe(sqe, &fakeNotifyMgr_);
                        if (ret) {
                            // 执行成功，移除该节点并更新succList节点信息
                            UpdateSubGraghs(subGraph, root, sqe.succList, rootQue);
                            continue;
                        }

                        failCnt++;  // 执行失败累加计数
                        if (failCnt == rootQue.size() + 1) {
                            break;  // 所有节点执行失败，退出
                        }
                    }
                }
            }
        }
    }
}

void FakeStreamMgr::Append(int streamId, FakeSqe sqe)
{
    lock_guard<mutex> lock(fakeStreamMutex_);
    if (sqeQueues_.find(streamId) == sqeQueues_.end()) {
        sqeQueues_.emplace(streamId, vector<FakeSqe>());
    }
    sqe.devId = rankId_;
    sqeQueues_[streamId].push_back(sqe);
}

void FakeStreamMgr::SaveInstr(int dieId, Hccl::CustomChannelInfoIn &instr)
{
    lock_guard<mutex> lock(fakeStreamMutex_);
    if (dieId >= DIE_NUM) {
        return;
    }
    auto ccuDataTmp = (Hccl::CcuDataTypeUnion)(instr.data.dataInfo.dataArray[0]);
    auto instrPtr = reinterpret_cast<Hccl::CcuRep::CcuInstr*>(ccuDataTmp.insinfo.resourceAddr);
    auto instrInfoSize = instr.data.dataInfo.dataLen;
    auto instrCnt = instrInfoSize / sizeof(Hccl::CcuRep::CcuInstr);

    ccuInstrData_[dieId].instrCnt = instrCnt;
    ccuInstrData_[dieId].instrData.resize(instrCnt);
    for (uint32_t i = 0; i < instrCnt; i++) {
        ccuInstrData_[dieId].instrData[i] = instrPtr[i];
    }
}

// 查找当前SQE节点应该放在图组的哪个下标
int FakeStreamMgr::FindGraphIndex(int streamId, int graphId)
{
    // 遍历数组查找当前节点应该在哪个图组位置
    for (int i = 0; i < graphGroups_.size(); i++) {
        map<int, map<int, FakeSqe>> graphMap = graphGroups_[i];
        if (graphMap.count(streamId) == 0) {
            // 这个下标还没有存放该卡的图
            return i;
        }

        // 判断是否当前图节点已满
        map<int, FakeSqe> graphs = graphMap[streamId];
        auto it = std::find_if(graphs.begin(), graphs.end(), [](const std::pair<int, FakeSqe>& elem) {
            return elem.second.isLastNode;
        });
        bool hasEndNode = it != graphs.end();

        // 这个下标中已经有该卡的图了(图未满时)，判断是否节点重复
        if (graphMap[streamId].count(graphId) == 0 && !hasEndNode) {
            // 节点不重复，可以放在这个下标
            return i;
        }
    }

    // 当前数组所有下标都放不了，需要新增数组元素
    return graphGroups_.size();
}

void FakeStreamMgr::AppendGraph(int streamId, int graphId, FakeSqe sqe)
{
    lock_guard<mutex> lock(fakeStreamMutex_);
    if (graphGroups_.size() == 0) {
        // 当前数组为空，新增一个下标
        graphGroups_.push_back(map<int, std::map<int, FakeSqe>>());
        graphGroups_[0][streamId][graphId] = sqe;
        return;
    }

    int index = FindGraphIndex(streamId, graphId);
    if (index >= graphGroups_.size()) {
        // 当前数组容量不足，新增一个下标
        graphGroups_.push_back(map<int, std::map<int, FakeSqe>>());
        graphGroups_[index][streamId][graphId] = sqe;
        return;
    }

    // 查找到了可以插入下标
    graphGroups_[index][streamId][graphId] = sqe;
}

void FakeStreamMgr::DestroyStream(int *streamId)
{
    lock_guard<mutex> lock(fakeStreamMutex_);
    sqeQueues_.erase(*streamId);
}

FakeNotifyMgr *FakeStreamMgr::GetFakeNotifyMgr()
{
     return &fakeNotifyMgr_;
}

uint8_t *FakeStreamMgr::GetSqBufferAddr()
{
    return sqBuffer_;
}

int FakeStreamMgr::GetSqHead(uint32_t sqId)
{
    if (sqHeadMap_.find(sqId) != sqHeadMap_.end()) {
        return sqHeadMap_[sqId];
    }
    return 0;
}

void FakeStreamMgr::UpdataSqHead(uint32_t sqId, int head)
{
    sqHeadMap_[sqId] = head;
}

void FakeNotifyMgr::Init(bool aicpuFlag, void *shmBase)
{
    aicpuFlag_ = aicpuFlag;
    notifyCnts_.clear();
    shmPub_ = reinterpret_cast<ShmPub *>(shmBase);
}

int *FakeNotifyMgr::CreateNotify(int rank)
{
    ShmPoolLock shmPoolLock;

    if ((shmPub_ == nullptr) || (shmPub_->stream.notifyIdGen >= MAX_NOTIFY_NUM)) {
        std::cout << "FakeNotifyMgr::CreateNotify invalid notifyIdGen!" << std::endl;
        return nullptr;
    }

    int curNotifyId = shmPub_->stream.notifyIdGen++;
    shmPub_->stream.notifyIds[curNotifyId] = curNotifyId;
    notifyRanks_[curNotifyId] = rank;
    return &(shmPub_->stream.notifyIds[curNotifyId]);
}

int FakeNotifyMgr::GetRankIdByNotifyId(int notifyId)
{
    if (notifyRanks_.find(notifyId) != notifyRanks_.end()) {
        return notifyRanks_[notifyId];
    }
    return 0;
}

bool FakeNotifyMgr::Record(const FakeSqe &sqe)
{
    int notifyId = sqe.notifyId;
    int notifyCnt = sqe.notifyCnt;
    
    ShmPoolLock shmPoolLock;

    if ((shmPub_ == nullptr) || (notifyId >= MAX_NOTIFY_NUM)) {
        throw std::exception();
    }

    shmPub_->stream.notifyCnts[notifyId] += notifyCnt;  // 从共享内存取notifyId解锁状态
    // notifyCnts_[notifyId] += notifyCnt;
    return true;
}

bool FakeNotifyMgr::Wait(FakeSqe &sqe)
{
    int notifyId = sqe.notifyId;
    int notifyCnt = sqe.notifyCnt;
    sqe.notifyCnt = 0;  // 第一次Wait节点执行后需置零

    ShmPoolLock shmPoolLock;

    if ((shmPub_ == nullptr) || (notifyId >= MAX_NOTIFY_NUM)) {
        throw std::exception();
    }

    bool ret = false;
    shmPub_->stream.notifyCnts[notifyId] -= notifyCnt;  // 从共享内存取notifyId解锁状态
    // notifyCnts_[notifyId] -= notifyCnt;
    if (shmPub_->stream.notifyCnts[notifyId] < 0) {
        ret = false;  // Record数量小于Wait数量
    } else {
        ret = true;  // Record数量大于或等于Wait数量
    }

    return ret;
}

void FakeNotifyMgr::DestroyNotify(int *notifyId)
{
}