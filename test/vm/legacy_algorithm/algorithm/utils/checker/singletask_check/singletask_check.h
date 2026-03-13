#ifndef HCCLV1_SINGLETASK_CHECK_H
#define HCCLV1_SINGLETASK_CHECK_H

#include <vector>

#include "llt_common.h"
#include "checker_data_slice.h"
#include "task_def.h"
#include <queue>

using namespace std;

namespace checker {

class SingleTaskCheck {
public:
    HcclResult CheckSlaveTaskQueue();
    HcclResult CheckTaskMem(TaskNodePtr dummyStart);
private:
    
    HcclResult CheckSingleSlice(RankId taskRank, u32 queueId, u32 taskId, const DataSlice& slice, RankId sliceRank);
    HcclResult CheckTwoSliceOverlap(RankId rank, u32 queueId, u32 taskId, const DataSlice& sliceA, const DataSlice& sliceB);

    HcclResult CheckSingleTaskMem(TaskNodePtr curTask);
    HcclResult CheckSingleCcuTaskMem(TaskNodePtr curTask);
    void AddChildrenToQueue(TaskNode *node, std::set<TaskNode *> &visitedNodes,
        std::queue<TaskNode *> &walkQue, std::set<TaskNode *> &simulatedNodes);
};
}

#endif
