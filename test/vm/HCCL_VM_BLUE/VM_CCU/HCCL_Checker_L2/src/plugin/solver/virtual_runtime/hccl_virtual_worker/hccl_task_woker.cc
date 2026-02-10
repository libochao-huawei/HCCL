#include <iostream>
#include "hccl_ipc.h"
#include "hccl_task_woker.h"
#include "hccl_common_macro.h"
#include "hccl_sim_world_pub.h"

using namespace HcclSim;

namespace VirtualRunTime {
int AdaptiveThreadPool::thread_cnt_ = 0;
const std::map<TaskType, const std::string> AdaptiveThreadPool::taskNames_ = {{TaskType::REDUCE_ADD, "reduce_add"},
    {TaskType::REDUCE_MIN, "reduce_min"},
    {TaskType::REDUCE_MAX, "reduce_max"},
    {TaskType::MEM_CPY, "mem_cpy"},
    {TaskType::NOTIFY_RECORD, "record"},
    {TaskType::NOTIFY_WAIT, "wait"}};

// 构造函数：根据CPU核心数初始化线程池
AdaptiveThreadPool::AdaptiveThreadPool(int min_threads, int max_threads)
    : min_threads_(min_threads > 0 ? min_threads : GetOptimalMinThreads()),
      max_threads_(max_threads > 0 ? max_threads : GetOptimalMaxThreads()), idle_threads_(0), active_threads_(0),
      tasks_processed_(0), total_task_time_(0), shutdown_(false)
{
    workers_.resize(max_threads_);
    queue_mutex_ = new std::mutex[max_threads_];
    condition_ = new std::condition_variable[max_threads_];
    // 创建初始工作线程
    for (int i = 0; i < min_threads_; ++i) {
        AddWorker(i);
    }

    // 启动监控线程
    monitor_thread_ = std::thread([this] { MonitorThread(); });

    std::cout << "ThreadPool initialized: " << min_threads_ << "-" << max_threads_
              << " threads (" << GetPhysicalCores() << " physical cores)\n\0";
}

void AdaptiveThreadPool::AddWorker(int threadId)
{
    AdaptiveThreadPool::thread_cnt_++;
    std::thread t([this, threadId] { return WorkerThread(threadId); });
    workers_[threadId] = ThreadInfo(std::move(t), true, threadId);
}

// 提交任务到线程池
void AdaptiveThreadPool::Submit(HcclTaskMetaData taskData, const HcclTaskReq& taskReq)
{
    if (shutdown_) {
        throw std::runtime_error("Submit on stopped TaskThreadPool");
    }
    // 获取任务类型
    auto type = GetTaskType(taskData);
    if (type == TaskType::TASK_INVALID) {
        throw std::runtime_error("Submit on invalid task type.");
    }
    // 根据dispatchId确定由哪个worker线程执行
    auto threadId = taskReq.dispatchId % max_threads_;
    {
        std::unique_lock<std::mutex> lock(queue_mutex_[threadId]);
        
        // 判断是否需要增加线程
        if (workers_[threadId].enable == false) {
            AddWorker(threadId);
        }

        std::cout << "Dispatch task " << AdaptiveThreadPool::taskNames_.at(type) << ", dispatchId= " << taskReq.dispatchId
                  << ", current workers= " << thread_cnt_ << ", threadId= " << threadId << std::endl;
        // 根据任务类型添加到合适的队列
        switch (type) {
            case TaskType::MEM_CPY:
                workers_[threadId].AddTask(type, TaskMemcpy, taskReq.taskCid, taskData);
                break;
            case TaskType::REDUCE_ADD:
                workers_[threadId].AddTask(type, TaskReduceAdd, taskReq.taskCid, taskData);
                break;
            case TaskType::REDUCE_MAX:
                workers_[threadId].AddTask(type, TaskReduceMax, taskReq.taskCid, taskData);
                break;
            case TaskType::REDUCE_MIN:
                workers_[threadId].AddTask(type, TaskReduceMin, taskReq.taskCid, taskData);
                break;
            case TaskType::NOTIFY_RECORD: {
                workers_[threadId].AddTask(type, TaskRecord, taskReq.taskCid, taskData);
                break;
            }
            case TaskType::NOTIFY_WAIT:
            {
                workers_[threadId].AddTask(type, TaskWait, taskReq.taskCid, taskData);
                break;
            }
            default:
                return;
        }
    }
    // 通知工作线程
    condition_[threadId].notify_one();
}

// 关闭线程池
void AdaptiveThreadPool::Shutdown()
{
    if (shutdown_.exchange(true)) {
        return;
    }
    for (int i = 0; i < max_threads_; ++i) {
        condition_[i].notify_all();
    }
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    for (auto& worker : workers_) {
        if (worker.thread.joinable()) {
            worker.thread.join();
        }
    }
}

// 获取线程池状态信息
std::string AdaptiveThreadPool::GetStatus() {
    std::ostringstream oss;
    oss << "Threads: " << thread_cnt_ << "/" << max_threads_
        << " (Active: " << active_threads_.load() << ", Idle: " << idle_threads_.load() << ")"
        << " | Processed Tasks: " << tasks_processed_.load()
        << " | Avg time: " << GetAverageTaskTime() << "ms"
        << " | Remain Tasks: " << GetRemainSize();
    return oss.str();
}

// 工作线程函数
HcclVmResult AdaptiveThreadPool::WorkerThread(int threadId) {
    while (true) {
        TaskDesc taskDesc;
        bool got_task = false;

        std::unique_lock<std::mutex> lock(queue_mutex_[threadId]);
        // 等待任务或关闭通知
        condition_[threadId].wait(lock, [this, threadId] {
            return shutdown_.load() || !workers_[threadId].task_queue_.empty();
        });
        if (shutdown_.load()) {
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
        // 任务出队列
        if (!workers_[threadId].task_queue_.empty()) {
            taskDesc = workers_[threadId].task_queue_.front();
            std::cout << "[Runtime] threadId : " << threadId <<" OUT taskCid: " << taskDesc.taskCid << " taskType : " << static_cast<int>(taskDesc.type) << std::endl;
            workers_[threadId].task_queue_.pop();
            lock.unlock();
            got_task = true;
        }
        
        if (got_task) {
            auto ret = taskDesc.task(taskDesc.data);
            tasks_processed_++;
            // 提交任务结果
            HcclTaskRsp taskResult{taskDesc.taskCid, ret};
            HCCLVM_CHK_RET(HcclIpcPushResponse(taskResult));
            printf("[Runtime] push a taskCid : %lu, type= %d, by threadId= %d \n", taskDesc.taskCid, static_cast<int>(taskDesc.type), threadId);
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 监控线程函数：输出线程池状态
void AdaptiveThreadPool::MonitorThread() {
    while (!shutdown_) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        // std::cout << "[ThreadPool] " << GetStatus() << std::endl;
    }
}

// 获取CPU物理核心数
int AdaptiveThreadPool::GetPhysicalCores() const {
    return std::thread::hardware_concurrency();
}

// 获取最优初始线程数
int AdaptiveThreadPool::GetOptimalMinThreads() const {
    return std::max(MIN_THREAD_NUM, static_cast<int>(GetPhysicalCores() * MIN_THREAD_COE));
}

// 获取最大线程数
int AdaptiveThreadPool::GetOptimalMaxThreads() const {
    return std::max(MAX_THREAD_NUM, static_cast<int>(GetPhysicalCores() * MAX_THREAD_COE));
}

// 获取平均任务处理时间
double AdaptiveThreadPool::GetAverageTaskTime() const {
    int processed = tasks_processed_.load();
    return processed > 0 ? total_task_time_ / static_cast<double>(processed) : 0.0;
}

// 获取队列中的任务数量
int AdaptiveThreadPool::GetRemainSize()
{
    int taskCnt = 0;
    for (int i = 0; i < workers_.size(); i++) {
        std::unique_lock<std::mutex> lock(queue_mutex_[i]);
        workers_[i].task_queue_.size();
    }
    return taskCnt;
}

TaskType AdaptiveThreadPool::GetTaskType(const HcclTaskMetaData &taskData)
{
    if (taskData.taskType == HccLTaskMetaType::MEM_CPY) {
        return TaskType::MEM_CPY;
    } else if (taskData.taskType == HccLTaskMetaType::REDUCE) {
        return static_cast<TaskType>(taskData.taskData.reduce.reduceOp);
    } else if (taskData.taskType == HccLTaskMetaType::NOTIFY_WAIT) {
        return TaskType::NOTIFY_WAIT;
    } else if (taskData.taskType == HccLTaskMetaType::NOTIFY_RECORD) {
        return TaskType::NOTIFY_RECORD;
    } else {
        return TaskType::TASK_INVALID;
    }
}
}