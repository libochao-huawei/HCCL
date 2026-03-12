#include "gtest/gtest.h"
#include <vector>
#include <iostream>
#include <string.h>
#include <memory>
#include <unistd.h>

#include "testcast_comm.h"

using namespace VirtualRunTime;

namespace VirtualRunTimeTest {
class VirRuntimeBroadcastTest : public testing::Test {
protected:
    virtual void SetUp()
    {
        std::cout << "VirRuntimeBroadcastTest set up." << std::endl;
    }

    virtual void TearDown()
    {
    }
};

class VRTBroadcastTest : public VRTOpBase {
public:
    VRTBroadcastTest(SimOpParam param) : VRTOpBase(param) {}
    /*
     *   Broadcast算子：将rank0的input buffer，平分N段，广播给所有rank的output buffer。
     *   
    */
    void GenerateTasks() {
        tasks_.resize(param_.rankNum);
        taskDescs_.resize(param_.rankNum);

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
        PrintData();
        // 所有 rank 的输出应该与 rank0 的输入相同
        auto* rank0In = reinterpret_cast<void*>(inputBuffers_[0]);
        for (uint32_t rank = 0; rank < param_.rankNum; ++rank) {
            auto* rankOut = reinterpret_cast<void*>(outputBuffers_[rank]);
            EXPECT_EQ(memcmp(rankOut, rank0In, bufferSize), 0)
                << "Rank " << rank << " output does not match rank 0 input";
        }
    }
};

TEST_F(VirRuntimeBroadcastTest, VirRuntimeTest_broadcast_4rank_2data_int32_ut1)
{
    SimOpParam param;
    param.rankNum = 4;
    param.dataCount = 2;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    VRTBroadcastTest bcTc(param);

    // 初始化各个rank input/output buffer
    bcTc.Init();

    // 根据算子类型，生成对应task队列
    bcTc.GenerateTasks();

    // 提交任务到执行进程
    AdaptiveThreadPool workers(2, 10);
    bcTc.SubmitTasks(workers);

    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }

    // 预期output
    bcTc.VerifyResults();
}

TEST_F(VirRuntimeBroadcastTest, VirRuntimeTest_broadcast_4rank_1024data_average_int32_ut1)
{
    SimOpParam param;
    param.sliceCnt = 64; // 每次搬移64个data
    param.rankNum = 4;
    param.dataCount = 1024;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    VRTBroadcastTest bcTc(param);

    // 初始化各个rank input/output buffer
    bcTc.Init();

    // 根据算子类型，生成对应task队列
    bcTc.GenerateTasks();

    // 提交任务到执行进程
    AdaptiveThreadPool workers(2, 10);
    bcTc.SubmitTasks(workers);
    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }

    // 预期output
    bcTc.VerifyResults();
}

TEST_F(VirRuntimeBroadcastTest, VirRuntimeTest_broadcast_4rank_4096data_noaverge_int32_ut1)
{
    SimOpParam param;
    param.sliceCnt = 70; // 每次搬移70个data，非均分，有尾块
    param.rankNum = 4;
    param.dataCount = 100;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    VRTBroadcastTest bcTc(param);

    // 初始化各个rank input/output buffer
    bcTc.Init();

    // 根据算子类型，生成对应task队列
    bcTc.GenerateTasks();

    // 提交任务到执行进程
    AdaptiveThreadPool workers(2, 10);
    bcTc.SubmitTasks(workers);
    // 等待任务执行完成
    while (workers.GetProcessedTaskNum() != bcTc.taskNum_) {
        sleep(1);
    }

    // 预期output
    bcTc.VerifyResults();
}

}