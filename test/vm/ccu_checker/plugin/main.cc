#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <nlohmann/json.hpp>
#include "storage_manager.h"
#include "task_utils.h"
#include "hccl_verifier.h"
#include "hccl_vm_log.h"

using json = nlohmann::json;

// 使用原子变量控制程序生命周期
std::atomic<bool> g_keep_running{true};

// --- 业务函数修正 ---
void RunChecker(const std::string& data_id) {
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    storage.SetDataId(data_id);

    // zhf-todo: 暂时注释，后续L1/L2可以考虑合并
    // storage.LoadCheckerParam();
    // storage.LoadMemLayout();
    storage.LoadHcclVmSynthesisData();
    storage.LoadHcclVmInstrData();
    storage.LoadHcclVmTaskMetaData();
    storage.Trans2CheckerParam();

    HcclSim::AllRankTaskQueues taskQueues;
    ConvertTaskQueue(taskQueues);

    HCCL_VM_INFO("[Worker] Task processed: {}", data_id);
    HcclResult ret;
    HcclSim::CheckerParam param = storage.GetCheckerParam();
    HcclCMDType cmdType = param.cmdType;
    HCCL_VM_INFO("[zhf][info] start check semantic ....: cmdType={}, reduceOp={}",
        static_cast<u32>(cmdType), static_cast<u32>(param.reduceType));
    switch (cmdType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            ret = CheckAllReduce(taskQueues, param.rankSize, param.dataType, param.dataCount, param.reduceType);
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            ret = CheckAllGather(taskQueues, param.rankSize, param.dataType, param.dataCount);
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            ret = CheckReduceScatter(taskQueues, param.rankSize, param.dataType, param.dataCount, param.reduceType);
            break;
        case HcclCMDType::HCCL_CMD_SEND:
            ret = CheckSend(taskQueues, param.rankSize, param.dataType, param.dataCount, param.srcRank, param.dstRank);
            break;
        case HcclCMDType::HCCL_CMD_RECEIVE:
            ret = CheckRecv(taskQueues, param.rankSize, param.dataType, param.dataCount, param.srcRank, param.dstRank);
            break;
        case HcclCMDType::HCCL_CMD_BROADCAST:
            ret = CheckBroadcast(taskQueues, param.rankSize, param.dataType, param.dataCount, param.root);
            break;
        case HcclCMDType::HCCL_CMD_REDUCE:
            ret = CheckReduce(taskQueues, param.rankSize, param.dataType, param.dataCount, param.reduceType, param.root);
            break;
        case HcclCMDType::HCCL_CMD_SCATTER:
            ret = CheckScatter(taskQueues, param.rankSize, param.dataType, param.dataCount, param.root);
            break;
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            ret = CheckBatchSendRecv(taskQueues, param.rankSize, param.dataType, param.dataCount);
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
            ret = CheckAllGatherV(taskQueues, param.rankSize, param.vDataDes);
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            ret = CheckReduceScatterV(taskQueues, param.rankSize, param.reduceType, param.vDataDes);
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            ret = CheckAll2All(taskQueues, param.rankSize, static_cast<HcclDataType>(param.all2AllDataDes.sendType), param.all2AllDataDes.sendCount);
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            ret = CheckAll2AllVC(taskQueues, param.rankSize, static_cast<HcclDataType>(param.all2AllDataDes.sendType), param.all2AllDataDes.sendCountMatrix);
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
            printf("CheckOpSemantics does not support HCCL_CMD_ALLTOALLV, using HCCL_CMD_ALLTOALLVV instead\n");
            ret = CheckAll2AllVC(taskQueues, param.rankSize, static_cast<HcclDataType>(param.all2AllDataDes.sendType), param.all2AllDataDes.sendCountMatrix);
            // ret = CheckAllToAllV(taskQueues, param.rankSize, param.dataType, param.dataCount);
            break;
        default:
            // 处理未知的 cmdType
            HCCL_VM_ERROR("Unknown HcclCMDType {}", static_cast<u32>(cmdType));
            ret = HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("Checker failed with error code {}", static_cast<u32>(ret));
    } else {
        HCCL_VM_INFO("Checker Success.");
    }
}

// --- 分发函数修正 ---
// 不再使用 exit(0)，而是通过标记位通知主线程
void ProcessCommand(const std::string& line) {
    try {
        auto j = json::parse(line);
        std::string action = j.value("action", "");
        auto payload = j.value("payload", json::object());

        if (action == "status") {
            std::string status = payload.value("status", "");
            HCCL_VM_INFO("[Checker] Status signal received. Status: {}", status);
            if (status != "finish") {
                return;
            }
            std::string data_id = payload.value("data_id", "");
            std::thread(RunChecker, data_id).detach();
        } 
        else if (action == "stop") {
            HCCL_VM_INFO("[Checker] Stop signal received. Initiating shutdown...");
            g_keep_running.store(false); // 仅仅修改标志位
        }
        // 其他 action...
    } catch (const std::exception& e) {
        std::cerr << "[Error] Command processing failed: " << e.what() << std::endl;
    }
}

int main() {
    LogConfig config;
    config.fileBaseName = "checker";
    config.consoleLevel = 0;
    InitLogger(config);

    HCCL_VM_INFO("[Checker] Plugin process active. Listening for commands...");

    // 主循环检查标志位
    std::string line;
    // 注意：std::getline 是阻塞的。如果 stop 指令后没有后续输入，
    // 循环会卡在 getline。但在插件管理场景下，发送完 stop 后通常会关闭管道，
    // 导致 getline 返回 false。
    while (g_keep_running.load() && std::getline(std::cin, line)) {
        if (line.empty()) continue;
        ProcessCommand(line);
    }
    
    HCCL_VM_INFO("[Checker] shutdown.");
    return 0; // 整个进程唯一的正常出口
}