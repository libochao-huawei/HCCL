#include <thread>     
#include <chrono>  
#include <gtest/gtest.h>
#include "hccl_sim_shm_manager.h"
#include "hccl_sim_data_defs.h"
#include <boost/interprocess/ipc/message_queue.hpp>

namespace ipc = boost::interprocess;
using namespace HcclSim;

// 测试夹具：初始化/清理共享内存
class SHMManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试前清理残留共享内存
        ipc::shared_memory_object::remove(SHM_NAME);
        ipc::message_queue::remove("runtime_0");
        for (int i = 0; i < 5; ++i) {
            ipc::message_queue::remove(("proxy_" + std::to_string(i)).c_str());
        }
    }

    void TearDown() override {
        // 测试后销毁共享内存和IPC队列
        SHMManager::DestroyAllModules();
        ipc::shared_memory_object::remove(SHM_NAME);
    }
};

// 测试1：共享内存初始化（创建模式）
TEST_F(SHMManagerTest, InitShm_CreateMode_Success) {
    EXPECT_NO_THROW(SHMManager::InitShm(true));
    EXPECT_TRUE(SHMManager::GetSegmentManager() != nullptr);
}

// 测试2：共享内存初始化（打开模式，未创建时失败）
TEST_F(SHMManagerTest, InitShm_OpenMode_NotFound) {
    ipc::shared_memory_object::remove(SHM_NAME);
    SHMManager::ResetGlobalState(); // 重置全局状态
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // 延长延迟，确保清理生效
    EXPECT_THROW(SHMManager::InitShm(false), std::runtime_error);
}

// 测试3：分配共享内存块（void类型）
TEST_F(SHMManagerTest, AllocateShmMemory_Success) {
    SHMManager::InitShm(true);
    void* addr = SHMManager::AllocateShmMemory(1024, "test_mem");
    EXPECT_NE(addr, nullptr);

    // 验证重复分配同一key失败
    EXPECT_THROW(SHMManager::AllocateShmMemory(1024, "test_mem"), std::runtime_error);
}

// 测试4：创建CheckOpParam模块
TEST_F(SHMManagerTest, CreateCheckOpParamModule_Success) {
    SHMManager::InitShm(true);
    CheckOpParam* module = SHMManager::CreateCheckOpParamModule();
    EXPECT_NE(module, nullptr);
    EXPECT_EQ(module->op_type, 0);
    EXPECT_EQ(module->op_id, 0);
    EXPECT_FALSE(module->is_completed);
}

// 测试5：创建ShmSimWorld模块
TEST_F(SHMManagerTest, CreateSimWorldModule_Success) {
    SHMManager::InitShm(true);
    ShmSimWorld* module = SHMManager::CreateSimWorldModule(4);
    EXPECT_NE(module, nullptr);
    EXPECT_EQ(module->devNum, 4);
    EXPECT_EQ(module->notifyIdGen, 0);
}

// 测试6：创建ProxyOffsetArray模块
TEST_F(SHMManagerTest, CreateProxyOffsetArrayModule_Success) {
    SHMManager::InitShm(true);
    ProxyOffsetArray* module = SHMManager::CreateProxyOffsetArrayModule(4);
    EXPECT_NE(module, nullptr);
    EXPECT_EQ(module->GetOffsetCount(), 4);

    // 测试写入和读取偏移量
    module->WriteOffset(2, 0x1000);
    EXPECT_EQ(module->ReadOffset(2), 0x1000);
    EXPECT_EQ(module->ReadOffset(5, 0xffff), 0xffff); // 超出范围返回默认值
}

// 测试7：创建TaskCollection模块并添加任务
TEST_F(SHMManagerTest, TaskCollection_AddAndGetTask_Success) {
    SHMManager::InitShm(true);
    TaskCollection* module = SHMManager::CreateTaskCollectionModule(10);
    EXPECT_NE(module, nullptr);
    EXPECT_EQ(module->maxSize, 10);
    EXPECT_EQ(module->indexGen, 0);

    // 添加任务
    HcclTaskMetaData task{};
    task.rankId = 1;
    task.taskType = HccLTaskMetaType::REDUCE;
    EXPECT_TRUE(SHMManager::AddTaskToCollection(module, task));
    EXPECT_EQ(module->indexGen, 1);

    // 获取任务
    HcclTaskMetaData out_task{};
    EXPECT_TRUE(SHMManager::GetTaskFromCollection(module, 0, out_task));
    EXPECT_EQ(out_task.rankId, 1);
    EXPECT_EQ(out_task.taskType, HccLTaskMetaType::REDUCE);
}

// 测试8：IPC队列初始化和销毁
TEST_F(SHMManagerTest, InitAndDestroyIpc_Success) {
    SHMManager::InitShm(true);
    HcclVmResult ret = SHMManager::InitIpc(3, sizeof(HcclTaskReq), sizeof(HcclTaskRsp));
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证队列存在
    EXPECT_NO_THROW(ipc::message_queue(ipc::open_only, "runtime_0"));
    EXPECT_NO_THROW(ipc::message_queue(ipc::open_only, "proxy_0"));
    EXPECT_NO_THROW(ipc::message_queue(ipc::open_only, "proxy_2"));

    // 销毁IPC
    ret = SHMManager::DestroyIpc(3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证队列已删除
    EXPECT_THROW(ipc::message_queue(ipc::open_only, "runtime_0"), ipc::interprocess_exception);
}

// 测试9：销毁不存在的IPC队列（无异常）
TEST_F(SHMManagerTest, DestroyIpc_NonExistentQueue_Success) {
    SHMManager::InitShm(true);
    HcclVmResult ret = SHMManager::DestroyIpc(2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试10：批量销毁所有模块
TEST_F(SHMManagerTest, DestroyAllModules_Success) {
    SHMManager::InitShm(true);
    SHMManager::CreateCheckOpParamModule();
    SHMManager::CreateSimWorldModule(2);
    SHMManager::CreateProxyBufferModule(1024);
    SHMManager::CreateTaskCollectionModule(5);
    SHMManager::CreateProxyOffsetArrayModule(2);

    EXPECT_NO_THROW(SHMManager::DestroyAllModules());
    // 修正：DestroyAllModules 后共享内存仍存在，ManagerMutex 也存在，不能判断 GetSegmentManager() 为空
    // 改为判断模块是否已销毁
    EXPECT_THROW(SHMManager::GetCheckOpParamModule(), std::runtime_error);
    EXPECT_THROW(SHMManager::GetSimWorldModule(), std::runtime_error);
    EXPECT_THROW(SHMManager::GetProxyBufferModule(), std::runtime_error);
    EXPECT_THROW(SHMManager::GetTaskCollectionModule(), std::runtime_error);
    EXPECT_THROW(SHMManager::GetProxyOffsetArrayModule(), std::runtime_error);
}