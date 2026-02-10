#ifndef TASK_VENTILATOR_H
#define TASK_VENTILATOR_H

#include <deque>
#include <atomic>
#include <thread>
#include "hccl_common_defs.h"
#include "thread_safe_map.h"

class TaskVentilator {
public:
    TaskVentilator(const TaskVentilator&) = delete;
    TaskVentilator& operator=(const TaskVentilator&) = delete;

    static TaskVentilator& GetInstance();

    void AddTaskCid(uint16_t streamId, HcclTaskCid taskCid);

private:
    TaskVentilator();
    ~TaskVentilator();

    void LaunchTask();

private:
    std::thread taskThread_;
    std::atomic<bool> stop_{false};

    ThreadSafeMap<uint16_t, std::deque<HcclTaskCid>> taskVentilatorMap_; // <streamId, queue<cid>>
};

#endif //TASK_VENTILATOR_H
