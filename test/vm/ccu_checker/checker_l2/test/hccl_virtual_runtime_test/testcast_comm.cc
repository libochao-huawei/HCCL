#include <vector>
#include <iostream>
#include <string.h>
#include <memory>
#include <unistd.h>
#include <map>

#include "com_stub.h"
#include "testcast_comm.h"

namespace VirtualRunTimeTest {
static const std::map<HcclDataType, size_t> g_dataTypeSize = {{HcclDataType::HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
    {HcclDataType::HCCL_DATA_TYPE_INT16, sizeof(int16_t)},
    {HcclDataType::HCCL_DATA_TYPE_INT32, sizeof(int32_t)},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, sizeof(uint8_t)},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, sizeof(uint16_t)},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, sizeof(uint32_t)},
    // {HcclDataType::HCCL_DATA_TYPE_FP16, sizeof(FP)},
    {HcclDataType::HCCL_DATA_TYPE_FP32, sizeof(float)}};

void VRTOpBase::Init()
{
    std::cout<<"Init start"<<std::endl;
    auto res = g_dataTypeSize.find(param_.dataType);
    if (res != g_dataTypeSize.end()) {
        elementSize_ = res->second;
    }
    // 分配输入/输出缓冲区
    size_t bufferSize = param_.dataCount * elementSize_;
    inputBuffers_.resize(param_.rankNum);
    outputBuffers_.resize(param_.rankNum);
    allBuffers_  = (void*)malloc(bufferSize * param_.rankNum * 2); // input + output buffer
    std::cout<<"zhf-init base addr= "<<std::hex<<(uint64_t)allBuffers_<<std::endl;
    RegisterOffset2AddrMap(reinterpret_cast<uint64_t>(allBuffers_)); // 注册内存基地址
    memset(allBuffers_, 0, bufferSize * param_.rankNum * 2);
    for (uint32_t i = 0; i < param_.rankNum; ++i) {
        inputBuffers_[i] = reinterpret_cast<uint64_t>(allBuffers_) + i * bufferSize;
        outputBuffers_[i] = reinterpret_cast<uint64_t>(allBuffers_) + (param_.rankNum + i) * bufferSize;
        std::cout<<"zhf-init input addr= "<<std::hex<<(uint64_t)inputBuffers_[i]<<", output addr= "<<std::hex<<(uint64_t)outputBuffers_[i]<<std::endl;
    }
    // 初始化 rank0 的输入数据
    if (param_.rankNum > 0) {
        for (uint32_t rank = 0; rank < param_.rankNum; ++rank) {
            auto* buffer = reinterpret_cast<int32_t*>(inputBuffers_[rank]);
            for (uint32_t i = 0; i < param_.dataCount; ++i) {
                buffer[i] = i + 1;  // 初始化值: 1, 2, 3, ...
            }
        }
    }
}
void VRTOpBase::SubmitTasks(AdaptiveThreadPool &workers) {
    // PrintData();
    std::cout<<"SubmitTasks start"<<std::endl;
    for (uint32_t rank = 0; rank < param_.rankNum; ++rank) {
        for (uint32_t idx = 0; idx < tasks_[rank].size(); idx++) {
            // 提交任务到virtual runtime
            workers.Submit(tasks_[rank][idx], taskDescs_[rank][idx]);
            taskNum_++;
        }
    }
}

void VRTOpBase::GenerateTaskCpyRank02RankX(uint32_t rank, uint32_t taskIdx, uint32_t lastCnt)
{
    std::cout << "GenerateTaskCpyRank02RankX rank" <<std::dec<< rank << ", idx= " << taskIdx << ", lastcnt=" << lastCnt
              << ", sliceCnt= " << param_.sliceCnt << ", eleSize= " << elementSize_ << std::endl;
    TransMemTask taskProc;
    uint64_t offsetWithInput = taskIdx * elementSize_ * param_.sliceCnt;
    uint64_t rank0InOffset   = inputBuffers_[0] - reinterpret_cast<uint64_t>(allBuffers_);
    taskProc.srcOffset = offsetWithInput + rank0InOffset;

    uint64_t rankOutOffset = outputBuffers_[rank] - reinterpret_cast<uint64_t>(allBuffers_);
    taskProc.dstOffset = offsetWithInput + rankOutOffset;
    auto dataCount = lastCnt == 0 ? param_.sliceCnt : lastCnt; // 尾块特殊处理
    taskProc.len = dataCount * elementSize_;

    std::cout << "GenerateTaskCpyRank02RankX data rank" << std::dec << rank << ", idx= " << taskIdx
              << ", offset= " << taskProc.srcOffset << ", dst offset= " << taskProc.dstOffset
              << ", dataCount= " << dataCount << std::endl;
    HcclTaskMetaData task;
    task.taskData.transMem = taskProc;
    task.taskType = HccLTaskMetaType::MEM_CPY;
    tasks_[rank].push_back(task);
    HcclTaskReq taskDesc;
    taskDesc.taskCid = (uint64_t)rank << 32 | taskIdx; // 暂时每个rank只有一个任务
    taskDesc.dispatchId = rank; // 暂时每个rank只有一个流
    taskDescs_[rank].push_back(taskDesc);
}

void VRTOpBase::GenerateTaskReduceRankX2Rank0(uint32_t rank, uint32_t taskIdx, uint32_t lastCnt)
{
    std::cout << "GenerateTaskReduceRankX2Rank0 rank" << rank << ", idx= " << taskIdx << ", lastcnt=" << lastCnt
              << ", sliceCnt= " << param_.sliceCnt << ", eleSize= " << elementSize_ << std::endl;
    ReduceTask taskProc;
    uint64_t offsetWithInput = taskIdx * elementSize_ * param_.sliceCnt;
    uint64_t rankInOffset    = inputBuffers_[rank] - reinterpret_cast<uint64_t>(allBuffers_);
    taskProc.srcOffset = offsetWithInput + rankInOffset;

    uint64_t rank0InOffset   = inputBuffers_[0] - reinterpret_cast<uint64_t>(allBuffers_);
    taskProc.dstOffset = offsetWithInput + rank0InOffset;

    taskProc.dataType = param_.dataType;
    taskProc.dataCount = lastCnt == 0 ? param_.sliceCnt : lastCnt; // 尾块特殊处理
    taskProc.reduceOp = param_.reduceOp;

    std::cout << "GenerateTaskReduceRankX2Rank0 data rank" << rank << ", idx= " << taskIdx
              << ", src offset= " << taskProc.srcOffset << ", dst offset= " << taskProc.dstOffset
              << ", dataCount= " << taskProc.dataCount << ", dataType= " << static_cast<int>(taskProc.dataType)
              << std::endl;
    HcclTaskMetaData task;
    task.taskData.reduce = taskProc;
    task.taskType = HccLTaskMetaType::REDUCE;
    tasks_[rank].push_back(task);
    HcclTaskReq taskDesc;
    taskDesc.taskCid = (uint64_t)rank << 32 | taskIdx; // 暂时每个rank只有一个任务
    taskDesc.dispatchId = rank; // 暂时每个rank只有一个流
    taskDescs_[rank].push_back(taskDesc);
}

void VRTOpBase::ResetTasks()
{
    for (uint32_t i = 0; i < param_.rankNum; ++i) {
        tasks_[i].clear();
        taskDescs_[i].clear();
    }
}

void VRTOpBase::PrintData()
{
    for (uint32_t rank = 0; rank < param_.rankNum; ++rank) {
        for (uint32_t idx = 0; idx < param_.dataCount; idx++) {
            auto in = reinterpret_cast<int32_t*>(inputBuffers_[0]);
            auto out = reinterpret_cast<int32_t*>(outputBuffers_[rank] );
            printf("cmp value idx=%d, input= %d: rank%d output= %d\n", idx, in[idx], rank, out[idx]);
        }
    }
}
}