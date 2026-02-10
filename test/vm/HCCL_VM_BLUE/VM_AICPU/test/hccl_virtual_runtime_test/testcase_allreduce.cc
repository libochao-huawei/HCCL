#include "gtest/gtest.h"
#include <vector>
#include <iostream>
#include <string.h>
#include <memory>
#include <unistd.h>

#include "testcast_comm.h"

using namespace VirtualRunTime;

namespace VirtualRunTimeTest {
class VirRuntimeAllreduceTest : public testing::Test {
protected:
    virtual void SetUp()
    {
        std::cout << "VirRuntimeAllreduceTest set up." << std::endl;
    }

    virtual void TearDown()
    {
    }
};

class VRTAllReduceTest : public VRTOpBase {
public:
    VRTAllReduceTest(SimOpParam param) : VRTOpBase(param) {}
    /*
     *   Allreduce算子：将所有rank的input buffer，做reduce后，广播给所有rank的output buffer。
     *   
    */
    void GenerateReduceTasks() {
        tasks_.resize(param_.rankNum);
        taskDescs_.resize(param_.rankNum);

        uint32_t times = 1;
        uint32_t tailCnt = param_.dataCount;
        if (param_.sliceCnt != 0) {
            times = (param_.dataCount + param_.sliceCnt - 1) / param_.sliceCnt;
            tailCnt = param_.dataCount % param_.sliceCnt;
        }

        // reduce任务: 所有rank的input buffer做reduce，汇聚到rank0 input buffer
        for (uint32_t rank = 1; rank < param_.rankNum; ++rank) {
            for (uint32_t idx = 0; idx < times; idx++) {
                if (idx == times - 1 && tailCnt != 0) {
                    GenerateTaskReduceRankX2Rank0(rank, idx, tailCnt);
                } else {
                    GenerateTaskReduceRankX2Rank0(rank, idx);
                }
            }
        }
        std::cout<<"generate "<<tasks_[0].size() + tasks_[1].size() + tasks_[2].size() + tasks_[3].size()<<" tasks in total."<<std::endl;
    }

    void GenerateBroadcastTasks() {
        uint32_t times = 1;
        uint32_t tailCnt = param_.dataCount;
        if (param_.sliceCnt != 0) {
            times = (param_.dataCount + param_.sliceCnt - 1) / param_.sliceCnt;
            tailCnt = param_.dataCount % param_.sliceCnt;
        }

        for (uint32_t rank = 0; rank < param_.rankNum; ++rank) {
            for (uint32_t idx = 0; idx < times; idx++) {
                if (idx == times - 1 && tailCnt != 0) {
                    GenerateTaskCpyRank02RankX(rank, idx, tailCnt);
                } else {
                    GenerateTaskCpyRank02RankX(rank, idx);
                }
            }
        }
    }

    void VerifyResults() {
        std::cout<<"VerifyResults start"<<std::endl;
        size_t bufferSize = param_.dataCount * elementSize_;
        // PrintData();
        // 所有 rank 的输出应该与 rank0 的输入相同
        auto* rank0In = reinterpret_cast<void*>(inputBuffers_[0]);
        for (uint32_t rank = 0; rank < param_.rankNum; ++rank) {
            auto* rankOut = reinterpret_cast<void*>(outputBuffers_[rank]);
            EXPECT_EQ(memcmp(rankOut, rank0In, bufferSize), 0)
                << "Rank " << rank << " output does not match rank 0 input";
        }
    }
};

TEST_F(VirRuntimeAllreduceTest, VirRuntimeTest_broadcast_4rank_2data_int32_ut1)
{
    SimOpParam param;
    param.rankNum = 4;
    param.dataCount = 2;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    param.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    VRTAllReduceTest bcTc(param);

    // 初始化各个rank input/output buffer
    bcTc.Init();

    // 根据算子类型，生成对应task队列
    bcTc.GenerateReduceTasks();

    // 提交任务到执行进程(reduce和boradcast需要分步提交&执行，实际业务通过post/wait同步)
    AdaptiveThreadPool workers(2, 10);
    bcTc.SubmitTasks(workers);
    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }
    // bcTc.PrintData();
    // 重置task队列（避免重复submit）
    bcTc.ResetTasks();

    bcTc.GenerateBroadcastTasks();
    bcTc.SubmitTasks(workers);

    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }
    // bcTc.PrintData();

    // 预期output
    bcTc.VerifyResults();
}

TEST_F(VirRuntimeAllreduceTest, VirRuntimeTest_broadcast_4rank_1024data_average_int32_ut1)
{
    SimOpParam param;
    param.rankNum = 4;
    param.sliceCnt = 64;
    param.dataCount = 1024;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    param.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    VRTAllReduceTest bcTc(param);

    // 初始化各个rank input/output buffer
    bcTc.Init();

    // 根据算子类型，生成对应task队列
    bcTc.GenerateReduceTasks();

    // 提交任务到执行进程(reduce和boradcast需要分步提交&执行，实际业务通过post/wait同步)
    AdaptiveThreadPool workers(2, 10);
    bcTc.SubmitTasks(workers);
    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }
    // bcTc.PrintData();
    // 重置task队列（避免重复submit）
    bcTc.ResetTasks();

    bcTc.GenerateBroadcastTasks();
    bcTc.SubmitTasks(workers);

    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }
    // bcTc.PrintData();

    // 预期output
    bcTc.VerifyResults();
}

TEST_F(VirRuntimeAllreduceTest, VirRuntimeTest_broadcast_4rank_4096data_noaverge_int32_ut1)
{
    SimOpParam param;
    param.rankNum = 4;
    param.sliceCnt = 70;
    param.dataCount = 100;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    param.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    VRTAllReduceTest bcTc(param);

    // 初始化各个rank input/output buffer
    bcTc.Init();

    // 根据算子类型，生成对应task队列
    bcTc.GenerateReduceTasks();

    // 提交任务到执行进程(reduce和boradcast需要分步提交&执行，实际业务通过post/wait同步)
    AdaptiveThreadPool workers(2, 10);
    bcTc.SubmitTasks(workers);
    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }
    // bcTc.PrintData();
    // 重置task队列（避免重复submit）
    bcTc.ResetTasks();

    bcTc.GenerateBroadcastTasks();
    bcTc.SubmitTasks(workers);

    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }
    // bcTc.PrintData();

    // 预期output
    bcTc.VerifyResults();
}

}