#ifndef HCCL_TESTCASE_COMMON_H
#define HCCL_TESTCASE_COMMON_H

#include <vector>
#include "hccl_common_defs.h"
#include "hccl_task_woker.h"
#include "hccl_task_thread.h"
#include "hccl_task_reduce_process.h"

using namespace VirtualRunTime;

namespace VirtualRunTimeTest {
struct SimOpParam {
    uint32_t sliceCnt{0}; // input buffer数据分段搬移，每一段为sliceCnt个data
    uint32_t rankNum;
    uint32_t dataCount;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

class VRTOpBase {
public:
    VRTOpBase(SimOpParam param) : param_(param) {}
    ~VRTOpBase() {
        free(allBuffers_);
    }
    void Init();
    void SubmitTasks(AdaptiveThreadPool &workers);
    void GenerateTaskCpyRank02RankX(uint32_t rank, uint32_t taskIdx, uint32_t lastCnt = 0);
    void GenerateTaskReduceRankX2Rank0(uint32_t rank, uint32_t taskIdx, uint32_t lastCnt = 0);
    void ResetTasks();
    void PrintData();

public:
    uint32_t taskNum_{0};
    SimOpParam param_;
    size_t elementSize_{0};
    void* allBuffers_;
    std::vector<uint64_t> inputBuffers_;
    std::vector<uint64_t> outputBuffers_;
    std::vector<std::vector<HcclTaskMetaData>> tasks_;
    std::vector<std::vector<HcclTaskReq>> taskDescs_;
};

}
#endif