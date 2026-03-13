#include <iostream>
#include <map>
#include <string.h>
#include "hccl_task_thread.h"
#include "hccl_task_reduce_process.h"
#include "hccl_shm_pub.h"
#include "hccl_common_macro.h"

using namespace HcclSim;

// 本地任务定义
namespace VirtualRunTime {
    HcclVmResult TransformAddr(uint64_t srcOffset, uint64_t dstOffset, char **src, char **dst)
    {
        void *srcAddr = nullptr;
        void *dstAddr = nullptr;
        HCCLVM_CHK_RET(GetAddrByOffset(srcOffset, &srcAddr));
        HCCLVM_CHK_RET(GetAddrByOffset(dstOffset, &dstAddr));
        HCCLVM_CHK_PTR(srcAddr);
        HCCLVM_CHK_PTR(dstAddr);
        std::cout<<"zhf-src addr= "<<srcAddr<<" zhf-dst addr= "<<dstAddr<<std::endl;

        *src = reinterpret_cast<char *>(srcAddr);
        *dst = reinterpret_cast<char *>(dstAddr);
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // Memory copy task
    HcclVmResult TaskMemcpy(const HcclTaskMetaData &task) {
        std::cout << "Memory copy task from src offset= " << task.taskData.transMem.srcOffset
                  << " to dst offset= " << task.taskData.transMem.dstOffset << " started\n";
        // 拷贝数据
        char *src = nullptr;
        char *dst = nullptr;
        HCCLVM_CHK_RET(TransformAddr(task.taskData.transMem.srcOffset, task.taskData.transMem.dstOffset, &src, &dst));
        memcpy(dst, src, task.taskData.transMem.len);
        std::cout << "Memory copy task completed.\n";
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // Reduce add task
    HcclVmResult TaskReduceAdd(const HcclTaskMetaData &task) {
        std::cout << "Reduce Add task from src offset= " << task.taskData.reduce.srcOffset
                  << " to dst offset= " << task.taskData.reduce.dstOffset << " started\n";
        // Reduce add
        char *src = nullptr;
        char *dst = nullptr;
        HCCLVM_CHK_RET(TransformAddr(task.taskData.reduce.srcOffset, task.taskData.reduce.dstOffset, &src, &dst));
        auto dataType = static_cast<HcclDataType>(task.taskData.reduce.dataType);
        MemReduceSum(src, dst, task.taskData.reduce.dataCount, dataType);

        std::cout << "Reduce add task completed.\n";
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // Reduce max task
    HcclVmResult TaskReduceMax(const HcclTaskMetaData &task) {
        std::cout << "Reduce Max task from src offset= " << task.taskData.reduce.srcOffset
                  << " to dst offset= " << task.taskData.reduce.dstOffset << " started\n";
        // Reduce max
        char *src = nullptr;
        char *dst = nullptr;
        HCCLVM_CHK_RET(TransformAddr(task.taskData.reduce.srcOffset, task.taskData.reduce.dstOffset, &src, &dst));
        auto dataType = static_cast<HcclDataType>(task.taskData.reduce.dataType);
        MemReduceSum(src, dst, task.taskData.reduce.dataCount, dataType);
        std::cout << "Reduce max task completed.\n";
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // Reduce min task
    HcclVmResult TaskReduceMin(const HcclTaskMetaData &task) {
        std::cout << "Reduce Min task from src offset= " << task.taskData.reduce.srcOffset
                  << " to dst offset= " << task.taskData.reduce.dstOffset << " started\n";
        // Reduce min
        char *src = nullptr;
        char *dst = nullptr;
        HCCLVM_CHK_RET(TransformAddr(task.taskData.reduce.srcOffset, task.taskData.reduce.dstOffset, &src, &dst));
        auto dataType = static_cast<HcclDataType>(task.taskData.reduce.dataType);
        MemReduceSum(src, dst, task.taskData.reduce.dataCount, dataType);
        std::cout << "Reduce min task completed.\n";
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    HcclVmResult TaskRecord(const HcclTaskMetaData &task)
    {
        bool result = false;
        uint64_t notify = task.taskData.notify.notifyId;
        HcclVmResult ret = SetNotifyValue(notify, true);
        if (ret != HcclVmResult::HcclVmResult::HCCL_SIM_SUCCESS) {
            std::cout << "TaskPost SetNotifyValue Failed\n";
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        std::cout << "Record " << notify << " task completed.\n";
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    HcclVmResult TaskWait(const HcclTaskMetaData &task)
    {
        uint64_t notify = task.taskData.notify.notifyId;
        
        bool result = false;
        while (true) {
            WaitNotifyValue(notify, &result);
            if (result) {
                break;
            }
            usleep(10 * 1000);
        }
        std::cout << "Wait " << notify << " task completed.\n";
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }
}