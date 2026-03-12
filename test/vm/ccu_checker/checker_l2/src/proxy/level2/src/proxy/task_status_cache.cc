#include <iostream>
#include "task_status_cache.h"
#include "hccl_ipc.h"
#include "hccl_vm_log.h"

using namespace std;
using namespace HcclSim;

TaskStatusCache& TaskStatusCache::GetInstance()
{
    static TaskStatusCache instance;
    return instance;
}

void TaskStatusCache::SetCurrentRankId(uint16_t rankId)
{
    rankId_ = rankId;
}

TaskStatusCache::TaskStatusCache()
{
    // 起线程从IPC查询result
    taskThread_ = std::thread(&TaskStatusCache::PullResult, this);
    taskThread_.detach();
}

void TaskStatusCache::PullResult()
{
    HCCL_VM_DEBUG("[ResultPuller] puller started");
    while (!stop_.load()) {
        HcclTaskRsp rsp;
        auto ret = HcclIpcPullResponse(rankId_, rsp);
        if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_DEBUG("[ResultPuller] TaskCid={}, Status={}", rsp.taskCid, rsp.status);
            taskStatusCache_.insert(rsp.taskCid, static_cast<HccLTaskStatus>(rsp.status));
        } else {
            HCCL_VM_WARN("[ResultPuller] HcclIpcPullResponse failed, ret={}", static_cast<uint32_t>(ret));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

TaskStatusCache::~TaskStatusCache()
{
    // 将线程停止
    if (!stop_.load()) {
        stop_.store(true);
    }
}

void TaskStatusCache::AddTaskCid(uint64_t streamId, HcclTaskCid taskCid)
{
    taskStreamMap_.insert(taskCid.value, streamId);
    taskStatusCache_.insert(taskCid.value, HccLTaskStatus::PENDING);
};

bool TaskStatusCache::IsStreamFinish(uint64_t streamId)
{
    bool isFinish = true;
    taskStreamMap_.for_each([&isFinish, streamId, this](uint64_t cid, uint64_t currentStreamId) {
        // 检查当前任务是否与指定的 streamId 相关联
        if (currentStreamId == streamId) {
            // 获取任务状态
            HccLTaskStatus status = taskStatusCache_[cid];
            // 如果任务未完成，则将 isFinish 设置为 false 并终止遍历
            if (status == HccLTaskStatus::PENDING) {
                isFinish = false;
                return; // 终止遍历
            }
        }
    });
    return isFinish;
}
