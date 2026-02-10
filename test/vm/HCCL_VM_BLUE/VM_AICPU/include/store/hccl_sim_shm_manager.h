#ifndef HCCLSIM_SHM_MANAGER_H
#define HCCLSIM_SHM_MANAGER_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>  
#include <boost/interprocess/allocators/allocator.hpp> 
#include <boost/interprocess/sync/interprocess_mutex.hpp> 
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <atomic>
#include <string>
#include <cstdint>
#include <stdexcept>

#include "hccl_sim_data_defs.h"

namespace ipc = boost::interprocess;

// -------------------------- 通用常量声明 --------------------------
extern const char* const SHM_NAME;
extern const char* const SHM_MUTEX_NAME;

// 模块名称常量声明
extern const char* const SHM_MODULE_CHECK_OP_PARAM;
extern const char* const SHM_MODULE_SIM_WORLD;
extern const char* const SHM_MODULE_PROXY_BUFFER;
extern const char* const SHM_MODULE_TASK_COLLECTION;
extern const char* const SHM_MODULE_PROXY_OFFSET_ARRAY;
extern const char* const SHM_MODULE_COMM_DOMAIN;
extern const char* const SHM_MODULE_MOCK_PROXY_BUFFER;
extern const char* const SHM_MODULE_NPU_IDX_MAP;
extern const char* const SHM_MODULE_MODE; 


// -------------------------- ProxyOffsetArray 结构体声明 --------------------------
struct ProxyOffsetArray {
    mutable ipc::interprocess_mutex mutex;
    ShmVector<uint64_t> data;

    ProxyOffsetArray(ShmSegmentManager* seg_mgr, size_t init_size = 0);

    void WriteOffset(size_t idx, uint64_t offset);
    uint64_t ReadOffset(size_t idx, uint64_t default_offset = 0) const;
    size_t GetOffsetCount() const;
    void ResizeOffsetArray(size_t new_count, uint64_t fill_offset = 0);

    ProxyOffsetArray() = delete;
    ProxyOffsetArray(const ProxyOffsetArray&) = delete;
    ProxyOffsetArray& operator=(const ProxyOffsetArray&) = delete;
};

struct ShmBufferPlaceholder {
    ipc::offset_ptr<void> buffer_ptr;
};

static const size_t SHM_DEFAULT_SIZE = 4ULL * 1024 * 1024 * 1024;  // 20GB
const size_t SHM_DEFAULT_BUFFER_SIZE = 1ULL * 1024 * 1024 * 1024;  // 10GB
const size_t SHM_TASK_COLLECTION_LENGTH = 20ULL * 1024 * 1024;  // 约20,000,000条
 

// -------------------------- SHMManager 类 --------------------------
class SHMManager {
public:
    static void InitShm(bool create, size_t size = SHM_DEFAULT_SIZE);
    static void DestroyShm();
    static void* AllocateShmMemory(uint64_t size, const std::string& memkey);
    static HcclSim::HcclVmResult InitIpc(uint32_t rankNum, size_t task_req_size, size_t task_rsp_size);
    static HcclSim::HcclVmResult DestroyIpc(uint32_t rankNum);

    template <class T, class... Args>
    static T* ConstructShmObject(const std::string& memkey, Args&&... args);

    template <class T>
    static T* FindShmObject(const std::string& memkey);

    // 仅保留主模板声明（删除类内所有 void 特化相关代码）
    template <class T>
    static bool DestroyShmObject(const std::string& memkey);

    static ShmSegmentManager* GetSegmentManager();
    static ipc::managed_shared_memory& GetSegment();

    // 业务模块接口
    static CheckOpParam* CreateCheckOpParamModule();
    static CheckOpParam* GetCheckOpParamModule();
    static ShmSimWorld* CreateSimWorldModule(const TopoMeta& topoMeta);
    static ShmSimWorld* GetSimWorldModule();
    static ProxyBuffer* CreateProxyBufferModule(size_t bufferSize);
    static ProxyBuffer* GetProxyBufferModule();
    static TaskCollection* CreateTaskCollectionModule(uint32_t max_task_count);
    static TaskCollection* GetTaskCollectionModule();
    static bool AddTaskToCollection(TaskCollection* task_collect, const HcclTaskMetaData& task);
    static uint32_t GetTaskCollectionCount(TaskCollection* task_collect);
    static bool GetTaskFromCollection(TaskCollection* task_collect, uint32_t idx, HcclTaskMetaData& out_task);
    static ProxyOffsetArray* CreateProxyOffsetArrayModule(size_t init_offset_count = 0);
    static ProxyOffsetArray* GetProxyOffsetArrayModule();    
    static void SetHcclVmMode(uint32_t mode);
    static uint32_t GetHcclVmMode();

    static void DestroyAllModules();

    static void ResetGlobalState();

private:
    static ipc::managed_shared_memory s_shm;
    static std::atomic_bool s_initialized;
    static uint32_t s_mode;

    static bool CreateSingleMessageQueue(const std::string& queue_name, size_t max_msg_count, size_t msg_size);
    static bool DestroySingleMessageQueue(const std::string& queue_name);

    SHMManager() = delete;
    ~SHMManager() = delete;
    SHMManager(const SHMManager&) = delete;
    SHMManager& operator=(const SHMManager&) = delete;
};

template <class T, class... Args>
T* SHMManager::ConstructShmObject(const std::string& memkey, Args&&... args) {
    if (!s_shm.get_segment_manager()) {
        SHMManager::InitShm(false);
        if (!s_shm.get_segment_manager()) {
            throw std::runtime_error("SHMManager: 共享内存未初始化");
        }
        // throw std::runtime_error("SHMManager: 共享内存未初始化");
    }

    auto *mutex = FindShmObject<ipc::interprocess_mutex>(SHM_MUTEX_NAME);
    if (!mutex) {
        throw std::runtime_error("SHMManager: 未找到HcclSimSharedMemoryMutex");
    }

    ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);
    if (FindShmObject<T>(memkey) != nullptr) {
        throw std::runtime_error("SHMManager: 共享对象'" + memkey + "'已存在");
    }

    return s_shm.construct<T>(memkey.c_str())(std::forward<Args>(args)...);
}

template <class T>
T* SHMManager::FindShmObject(const std::string& memkey) {
    if (!s_shm.get_segment_manager()) {
        SHMManager::InitShm(false);
        if (!s_shm.get_segment_manager()) {
            throw std::runtime_error("SHMManager: 共享内存未初始化");
            return nullptr;
        }
    }
    auto ret = s_shm.find<T>(memkey.c_str());
    return ret.first;
}

// 主模板（非 void 类型）
template <class T>
inline bool SHMManager::DestroyShmObject(const std::string& memkey) {
    if (!s_shm.get_segment_manager()) {
        SHMManager::InitShm(false);
        if (!s_shm.get_segment_manager()) {
            throw std::runtime_error("SHMManager: 共享内存未初始化");
        }
    }
    auto* mutex = FindShmObject<ipc::interprocess_mutex>(SHM_MUTEX_NAME);
    if (!mutex) {
        throw std::runtime_error("SHMManager: 未找到HcclSimSharedMemoryMutex");
    }
    ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);
    return s_shm.destroy<T>(memkey.c_str());
}

// 类外 void 特化模板
template <>
inline bool SHMManager::DestroyShmObject<void>(const std::string& memkey) {
    if (!s_shm.get_segment_manager()) {
        SHMManager::InitShm(false);
        if (!s_shm.get_segment_manager()) {
            throw std::runtime_error("SHMManager: 共享内存未初始化");
        }
    }
    auto* mutex = FindShmObject<ipc::interprocess_mutex>(SHM_MUTEX_NAME);
    if (!mutex) {
        throw std::runtime_error("SHMManager: 未找到HcclSimSharedMemoryMutex");
    }
    ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);

    // 步骤1：查找占位符
    ShmBufferPlaceholder* placeholder = FindShmObject<ShmBufferPlaceholder>(memkey);
    if (!placeholder) {
        return false;
    }

    // 步骤2：释放实际缓冲区内存
    s_shm.deallocate(placeholder->buffer_ptr.get());

    // 步骤3：销毁占位符对象
    s_shm.destroy<ShmBufferPlaceholder>(memkey.c_str());

    return true;
}

#endif