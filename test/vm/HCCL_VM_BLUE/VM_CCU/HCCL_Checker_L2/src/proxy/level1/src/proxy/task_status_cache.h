#ifndef TASK_STATUS_CACHE_H
#define TASK_STATUS_CACHE_H

#include <thread>
#include <atomic>
#include "thread_safe_map.h"
#include "hccl_common_defs.h"

class TaskStatusCache {
public:
    TaskStatusCache(const TaskStatusCache&) = delete;
    TaskStatusCache& operator=(const TaskStatusCache&) = delete;

    static TaskStatusCache& GetInstance();

    void SetCurrentRankId(uint16_t rankId);

    void AddTaskCid(uint64_t streamId, HcclTaskCid taskCid);
    bool IsStreamFinish(uint64_t streamId);

private:
    TaskStatusCache();
    ~TaskStatusCache();

    void PullResult();

private:
    uint16_t rankId_;
    std::thread taskThread_;
    std::atomic<bool> stop_{false};

    ThreadSafeMap<uint64_t, uint64_t> taskStreamMap_; // <cid, streamId>
    ThreadSafeMap<uint64_t, HccLTaskStatus> taskStatusCache_; // <cid, status>
};

#endif //TASK_STATUS_CACHE_H
