#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include "hccl_common_defs.h"
#include "hccl_task_woker.h"
#include "hccl_shm_pub.h"
#include "hccl_ipc.h"
#include "hccl_common_macro.h"
#include <nlohmann/json.hpp>

uint16_t g_rankId = 0;

HcclSim::HcclVmResult TaskFetcherLoop(VirtualRunTime::AdaptiveThreadPool* workers) {
    while (true) {
        // pull task desc
        HcclTaskReq taskDesc;
        HCCLVM_CHK_RET(HcclIpcPullRequest(taskDesc));
        // get task data
        HcclTaskMetaData taskData;
        HCCLVM_CHK_RET(GetTaskCollectionByCid(taskDesc.taskCid, &taskData));
        // submit task
        workers->Submit(taskData, taskDesc);
    }
}

int main(int argc, char* argv[])
{
    g_hcclComm.SetIpcMode(1);
    SHMManager::SetHcclVmMode(1);
    uint32_t mode = SHMManager::GetHcclVmMode();
    printf("[GetHcclVmMode] mode: %u\n", mode);
    VirtualRunTime::AdaptiveThreadPool workers(2, 0);
    std::thread fetcherThread(TaskFetcherLoop, &workers);

    std::cout << "[Runner] Plugin process active. Listening for commands..." << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        try {
            auto j = nlohmann::json::parse(line);
            std::string action = j.value("action", "");

            if (action == "stop") {
                std::cout << "[Runner] Stop action detected. Starting shutdown..." << std::endl;
                break; // 跳出循环，准备清理
            }
            // 处理其他 action...
        } catch (...) {
            std::cerr << "Invalid command: " << line << std::endl;
            continue;
        }
    }

    workers.Shutdown();

    exit(0);

    // todo: 后续最小和最大线程数，可以根据用例配置估算
    return 0;
}