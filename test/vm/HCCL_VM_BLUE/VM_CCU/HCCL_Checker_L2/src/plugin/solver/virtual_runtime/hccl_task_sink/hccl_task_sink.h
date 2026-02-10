#ifndef HCCL_TASK_SINK_H
#define HCCL_TASK_SINK_H

#include "hccl_common_defs.h"

#include <thread>
#include <vector>

class HcclTaskSink {
public:
    HcclTaskSink();
    ~HcclTaskSink();
    bool Init(int threadNum = 1);
private:
    void RunThr(int threadIdx);
private:
    std::vector<std::thread*> m_threads;
};

#endif