#ifndef HCCL_TASK_WORKER_H
#define HCCL_TASK_WORKER_H

#include <map>
#include <vector>
#include <future>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <queue>
#include <thread>
#include <mutex>
#include <functional>
#include <atomic>
#include <condition_variable>
#include "hccl_task_thread.h"
#include <list>

namespace VirtualRunTime {
#define MIN_THREAD_NUM 2 // 默认最小线程数2
#define MAX_THREAD_NUM 16 // 默认最大线程数16
#define MIN_THREAD_COE 0.75 // 根据硬件并行数计算最小线程系数
#define MAX_THREAD_COE 4.0 // 根据硬件并行数计算最大线程系数

using workerFunc = std::function<HcclSim::HcclVmResult(const HcclTaskMetaData&)>;

// 任务类型枚举
enum class TaskType {
    NOTIFY_RECORD,
    NOTIFY_WAIT,
    REDUCE_ADD,
    REDUCE_MIN,
    REDUCE_MAX,
    MEM_CPY, // 内存拷贝
    TASK_INVALID
};

// 任务描述结构
struct TaskDesc {
    TaskType type{TaskType::TASK_INVALID};
    HcclTaskMetaData data{};
    workerFunc task;
    uint64_t taskCid;

    TaskDesc() = default;
    TaskDesc(const TaskDesc &taskDesc) {
        this->type    = taskDesc.type;
        this->task    = taskDesc.task;
        this->data    = taskDesc.data;
        this->taskCid = taskDesc.taskCid;
    }
    TaskDesc(TaskType t, workerFunc f, uint64_t cid, const HcclTaskMetaData &d)
        : type(t), task(f), taskCid(cid), data(d)
    {}
};

struct ThreadInfo {
    std::thread thread;
    bool enable{false};
    int workerId{0};
    std::queue<TaskDesc> task_queue_; // 任务队列

    ThreadInfo() = default;
    ThreadInfo(std::thread&& t, bool e, int id) : thread(std::move(t)), enable(e), workerId(id) {}

    void AddTask(TaskType type, workerFunc f, uint64_t cid, const HcclTaskMetaData &data)
    {
        task_queue_.emplace(type, f, cid, data);
    }
};

class AdaptiveThreadPool {
public:
    // 构造函数：根据CPU核心数初始化线程池
    AdaptiveThreadPool(int min_threads = -1, int max_threads = -1);

    // 析构函数：优雅关闭线程池
    ~AdaptiveThreadPool() {
        Shutdown();
        delete[] queue_mutex_;
    }

    // 提交任务到线程池
    void Submit(HcclTaskMetaData taskData, const HcclTaskReq& taskReq);

    // 关闭线程池
    void Shutdown();

    // 获取线程池状态信息
    std::string GetStatus();

    // 获取已处理的任务数
    uint32_t GetProcessedTaskNum()
    {
        return tasks_processed_.load();
    }

private:
    // 工作线程函数
    HcclSim::HcclVmResult WorkerThread(int threadId);

    // 监控线程函数：输出线程池状态
    void MonitorThread();

    // 获取CPU物理核心数
    int GetPhysicalCores() const;

    // 获取最优初始线程数
    int GetOptimalMinThreads() const;

    // 获取最大线程数
    int GetOptimalMaxThreads() const;

    // 获取平均任务处理时间
    double GetAverageTaskTime() const;

    // 获取队列中的任务数量
    int GetRemainSize();

    // 新增worker线程
    void AddWorker(int threadId);

    TaskType GetTaskType(const HcclTaskMetaData &taskData);

private:
    static int thread_cnt_;
    static int queue_cnt_;
    const static std::map<TaskType, const std::string> taskNames_;

    // 线程池核心参数
    int min_threads_;
    int max_threads_;
    
    // 工作线程集合
    std::vector<ThreadInfo> workers_;
    
    // 监控线程
    std::thread monitor_thread_;
    
    // 同步原语
    std::mutex *queue_mutex_; // 任务队列:队列锁:工作线程 = 1:1:1
    std::condition_variable *condition_;
    
    // 状态统计
    std::atomic<int> idle_threads_;
    std::atomic<int> active_threads_;
    std::atomic<int> tasks_processed_;
    std::atomic<long> total_task_time_;
    std::atomic<bool> shutdown_;
};
}
#endif