#include <thread>     
#include <chrono>    
#include <gtest/gtest.h>
#include "hccl_shm_pub.h"
#include "hccl_sim_shm_manager.h"
#include "hccl_common_defs.h"

// 测试夹具：初始化共享内存模块
class ShmPubTest : public ::testing::Test {
protected:
    TopoMeta CreateTestTopoMeta() {
        // 构造测试拓扑：1个SuperPod → 1个Server → 2个设备
        TopoMeta topo;
        SuperPodMeta super_pod;
        ServerMeta server;
        server.push_back(0);
        server.push_back(1);
        super_pod.push_back(server);
        topo.push_back(super_pod);
        return topo;
    }

    void SetUp() override {
        // 步骤1：强制清理共享内存文件（彻底删除残留）
        ipc::shared_memory_object::remove(SHM_NAME);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));  // 等待系统清理完成

        // 步骤2：重置 SHMManager 的全局状态（关键！解决 s_initialized 残留）
        SHMManager::ResetGlobalState();  // 新增静态方法，手动重置 s_initialized

        // 步骤3：初始化共享内存（此时 s_initialized 为 false，能正常创建 ManagerMutex）
        TopoMeta topo = CreateTestTopoMeta();
        EXPECT_EQ(InitSharedMemory(topo), HCCL_SUCCESS);
    }

    void TearDown() override {
        SHMManager::DestroyAllModules();
        ipc::shared_memory_object::remove(SHM_NAME);
    }
};

// 测试1：InitSharedMemory（初始化所有模块）
TEST_F(ShmPubTest, InitSharedMemory_Success) {
    EXPECT_NE(SHMManager::GetCheckOpParamModule(), nullptr);
    EXPECT_NE(SHMManager::GetSimWorldModule(), nullptr);
    EXPECT_NE(SHMManager::GetProxyBufferModule(), nullptr);
    EXPECT_NE(SHMManager::GetTaskCollectionModule(), nullptr);
    EXPECT_NE(SHMManager::GetProxyOffsetArrayModule(), nullptr);
}

// 测试2：GetAddrByOffset（地址偏移计算）
TEST_F(ShmPubTest, GetAddrByOffset_Success) {
    void* base = SHMManager::GetProxyBufferModule();
    void* addr = nullptr;

    // 正常偏移
    EXPECT_EQ(GetAddrByOffset(0x100, &addr), HCCL_SUCCESS);
    EXPECT_EQ(addr, reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(base) + 0x100));

    // 偏移为0
    EXPECT_EQ(GetAddrByOffset(0, &addr), HCCL_SUCCESS);
    EXPECT_EQ(addr, base);

    // 无效参数（addr为空）
    EXPECT_EQ(GetAddrByOffset(0x100, nullptr), HCCL_E_PARA);

    // 偏移超出缓冲区大小
    EXPECT_EQ(GetAddrByOffset(SHM_DEFAULT_BUFFER_SIZE + 1, &addr), HCCL_E_PARA);
}

// 测试3：InsertTaskToCollection（插入任务）
TEST_F(ShmPubTest, InsertTaskToCollection_Success) {
    HcclTaskMetaData task{};
    task.rankId = 0;
    task.taskType = HccLTaskMetaType::MEM_CPY;
    uint32_t index = 0;

    // 插入任务
    EXPECT_EQ(InsertTaskToCollection(&task, &index), HCCL_SUCCESS);
    EXPECT_EQ(index, 0);

    // 插入第二个任务
    task.rankId = 1;
    EXPECT_EQ(InsertTaskToCollection(&task, &index), HCCL_SUCCESS);
    EXPECT_EQ(index, 1);

    // 无效参数（task为空）
    EXPECT_EQ(InsertTaskToCollection(nullptr, &index), HCCL_E_PARA);
}

// 测试4：GetTaskCollectionByCid（通过CID获取任务）
TEST_F(ShmPubTest, GetTaskCollectionByCid_Success) {
    // 先插入任务
    HcclTaskMetaData task{};
    task.rankId = 2;
    task.taskType = HccLTaskMetaType::REDUCE;
    uint32_t index = 0;
    InsertTaskToCollection(&task, &index);

    // 构造CID（commId=10, rankId=2, index=0）
    HcclTaskCid cid{};
    cid.field.commId = 10;
    cid.field.rankId = 2;
    cid.field.index = 0;

    // 获取任务
    HcclTaskMetaData out_task{};
    EXPECT_EQ(GetTaskCollectionByCid(cid.value, &out_task), HCCL_SUCCESS);
    EXPECT_EQ(out_task.rankId, 2);
    EXPECT_EQ(out_task.taskType, HccLTaskMetaType::REDUCE);

    // 无效参数（task为空）
    EXPECT_EQ(GetTaskCollectionByCid(cid.value, nullptr), HCCL_E_PTR);

    // 无效索引（超出范围）
    cid.field.index = 100;
    EXPECT_EQ(GetTaskCollectionByCid(cid.value, &out_task), HCCL_SIM_SHM_TASK_COLLECTION_ERROR);
}

// 测试5：AllocRankMemForProxy（分配Proxy内存）
TEST_F(ShmPubTest, AllocRankMemForProxy_Success) {
    void* addr1 = nullptr;
    void* addr2 = nullptr;

    // 分配内存
    EXPECT_EQ(AllocRankMemForProxy(1024, &addr1), HCCL_SUCCESS);
    EXPECT_NE(addr1, nullptr);

    // 分配第二个内存块（地址连续）
    EXPECT_EQ(AllocRankMemForProxy(2048, &addr2), HCCL_SUCCESS);
    EXPECT_EQ(addr2, reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(addr1) + 1024));

    // 无效参数（size=0）
    EXPECT_EQ(AllocRankMemForProxy(0, &addr1), HCCL_E_PARA);
}

// 测试6：InsertProxyBaseAddr（写入Proxy基地址）
TEST_F(ShmPubTest, InsertProxyBaseAddr_Success) {
    uint32_t rankId = 0;
    EXPECT_EQ(InsertProxyBaseAddr(rankId), HCCL_SUCCESS);

    // 验证写入的基地址
    ProxyOffsetArray* offset_array = SHMManager::GetProxyOffsetArrayModule();
    void* base = SHMManager::GetProxyBufferModule();
    EXPECT_EQ(offset_array->ReadOffset(rankId), reinterpret_cast<uint64_t>(base));

    // 无效rankId（超出MAX_DEV_NUM）
    EXPECT_EQ(InsertProxyBaseAddr(MAX_DEV_NUM + 1), HCCL_E_PARA);
}

// 测试7：GetRankOffset（计算地址偏移量）
TEST_F(ShmPubTest, GetRankOffset_Success) {
    uint32_t rankId = 0;
    void* base = SHMManager::GetProxyBufferModule();
    void* addr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(base) + 0x2000);

    // 先写入基地址
    InsertProxyBaseAddr(rankId);

    // 计算偏移量
    uint64_t offset = 0;
    EXPECT_EQ(GetRankOffset(rankId, addr, &offset), HCCL_SUCCESS);
    EXPECT_EQ(offset, 0x2000);

    // 地址小于基地址（无效）
    void* invalid_addr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(base) - 0x100);
    EXPECT_EQ(GetRankOffset(rankId, invalid_addr, &offset), HCCL_SIM_SHM_PROXY_OUT_OF_RANGE);
}

// 测试8：GetTaskCollection（获取所有任务）
TEST_F(ShmPubTest, GetTaskCollection_Success) {
    // 插入2个任务
    HcclTaskMetaData task1{};
    task1.rankId = 0;
    HcclTaskMetaData task2{};
    task2.rankId = 1;
    uint32_t index = 0;
    InsertTaskToCollection(&task1, &index);
    InsertTaskToCollection(&task2, &index);

    // 获取所有任务
    HcclTaskMetaData tasks[2]{};
    EXPECT_EQ(GetTaskCollection(tasks), HCCL_SUCCESS);
    EXPECT_EQ(tasks[0].rankId, 0);
    EXPECT_EQ(tasks[1].rankId, 1);
}

// 测试9：InitIpc（初始化IPC队列）
TEST_F(ShmPubTest, InitIpc_Success) {
    EXPECT_EQ(InitIpc(2), HCCL_SUCCESS);
    EXPECT_NO_THROW(ipc::message_queue(ipc::open_only, "runtime_0"));
    EXPECT_NO_THROW(ipc::message_queue(ipc::open_only, "proxy_1"));
}