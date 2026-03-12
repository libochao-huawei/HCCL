#include <thread>
#include <iostream>
#include "task_ventilator.h"
#include "hccl_ipc.h"

using namespace std;
using namespace HcclSim;

TaskVentilator& TaskVentilator::GetInstance()
{
    static TaskVentilator instance;
    return instance;
}

TaskVentilator::TaskVentilator()
{
    // 起线程下发Task到IPC
    taskThread_ = std::thread(&TaskVentilator::LaunchTask, this);
    taskThread_.detach();
}

void TaskVentilator::LaunchTask()
{
    cout << "Task ventilator started." << endl;
    while (!stop_.load()) {
        // demo直接将Task下发, 先不做判断
        taskVentilatorMap_.for_each([](uint16_t streamId, std::deque<HcclTaskCid>& taskCidQueue) {
            while (!taskCidQueue.empty()) {
                HcclTaskCid taskCid = taskCidQueue.front();
                uint64_t temp = taskCid.value;
                temp = temp >> 16;
                uint16_t rankId = 0xffff & temp;
                HcclTaskReq req{};
                req.taskCid = taskCid.value;
                req.dispatchId = rankId;
                // todo demo暂时不用填 req.dispatchId
                auto ret = HcclIpcPushRequest(req);
                if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
                    // 下发到IPC失败, 不再继续下发本stream队列的Task, 等线程的下次循环继续
                    cout << "[Warning]"
                         << "[TaskVentilator] HcclIpcPushRequest failed, ret=" << ret << " taskCid=" << taskCid.value
                         << endl;
                    break;
                }
                // 下发到IPC成功, TaskCid出队
                cout << "[TaskVentilator] Task Launched, cid=" << taskCid.value << endl;
                taskCidQueue.pop_front();
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

TaskVentilator::~TaskVentilator() {
    // 将线程停止
    if (!stop_.load()) {
        // set stop flag
        stop_.store(true);
        // wait thread stop
        // taskThread_.join();
        // cout << "Task ventilator stopped." << endl;
    }
}

void TaskVentilator::AddTaskCid(uint16_t streamId, HcclTaskCid taskCid)
{
    taskVentilatorMap_[streamId].push_back(taskCid);
}
