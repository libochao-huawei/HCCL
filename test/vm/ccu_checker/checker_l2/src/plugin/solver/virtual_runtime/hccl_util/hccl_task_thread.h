#ifndef HCCL_TASK_THREAD_H
#define HCCL_TASK_THREAD_H

#include <thread>
#include <vector>
#include <functional>
#include "hccl_common_defs.h"

namespace VirtualRunTime {
HcclSim::HcclVmResult TaskMemcpy(const HcclTaskMetaData &task);
HcclSim::HcclVmResult TaskReduceAdd(const HcclTaskMetaData &task);
HcclSim::HcclVmResult TaskReduceMax(const HcclTaskMetaData &task);
HcclSim::HcclVmResult TaskReduceMin(const HcclTaskMetaData &task);
HcclSim::HcclVmResult TaskRecord(const HcclTaskMetaData &task);
HcclSim::HcclVmResult TaskWait(const HcclTaskMetaData &task);
}

#endif