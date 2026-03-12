#include <store/hccl_sim_world_pub.h>
#include <store/hccl_sim_shm_manager.h>
#include <boost/interprocess/errors.hpp>
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include "hccl_vm_log.h"

using namespace HcclSim;

// -------------------------- 通用常量定义 --------------------------
const char *const SHM_NAME = "HcclSimSharedMemory";
const char *const SHM_MUTEX_NAME = "HcclSimSharedMemoryMutex";

// 模块名称常量定义
const char *const SHM_MODULE_CHECK_OP_PARAM = "CheckOpParam";
const char *const SHM_MODULE_SIM_WORLD = "SimWorld";
const char *const SHM_MODULE_PROXY_BUFFER = "ProxyBuffer";
const char *const SHM_MODULE_TASK_COLLECTION = "TaskCollection";
const char *const SHM_MODULE_PROXY_OFFSET_ARRAY = "ProxyOffsetArray";
const char *const SHM_MODULE_COMM_DOMAIN = "CommDomain";
const char *const SHM_MODULE_MOCK_PROXY_BUFFER = "MockProxyBuffer";
const char* const SHM_MODULE_NPU_IDX_MAP = "NpuPos2IndexMap";
const char* const SHM_MODULE_MODE = "HcclVmMode";
const char* const SHM_MODULE_PROXY_CONFIG = "ProxyConfig";

// -------------------------- 静态成员初始化 --------------------------
ipc::managed_shared_memory SHMManager::s_shm;
std::atomic_bool SHMManager::s_initialized(false);
uint32_t SHMManager::s_mode{0};

// -------------------------- CheckOpParam 构造函数实现 --------------------------
CheckOpParam::CheckOpParam(ShmSegmentManager *seg_mgr)
    : op_type(0), op_id(0), is_completed(false), op_desc(ShmStringAllocator(seg_mgr))
{}

// -------------------------- ProxyOffsetArray 结构体实现 --------------------------
ProxyOffsetArray::ProxyOffsetArray(ShmSegmentManager *seg_mgr, size_t init_size) : data(ShmAllocator<uint64_t>(seg_mgr))
{  // 正确：用 ShmAllocator 包装 seg_mgr
    if (init_size > 0) {
        data.resize(init_size, 0);
    }
}

void ProxyOffsetArray::WriteOffset(size_t idx, uint64_t offset)
{
    ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
    if (idx >= data.size()) {
        data.resize(idx + 1, 0);
    }
    data[idx] = offset;
}

uint64_t ProxyOffsetArray::ReadOffset(size_t idx, uint64_t default_offset) const
{
    ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
    if (idx >= data.size()) {
        return default_offset;
    }
    return data[idx];
}

size_t ProxyOffsetArray::GetOffsetCount() const
{
    ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
    return data.size();
}

void ProxyOffsetArray::ResizeOffsetArray(size_t new_count, uint64_t fill_offset)
{
    ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
    data.resize(new_count, fill_offset);
}

// -------------------------- SHMManager 基础接口实现 --------------------------
// 辅助函数：检查共享内存是否存在
static bool IsShmExists(const std::string& shm_name) {
    try {
        // 尝试以只读方式打开共享内存，成功则存在，失败则不存在
        ipc::shared_memory_object shm(ipc::open_only, shm_name.c_str(), ipc::read_only);
        return true;
    } catch (const ipc::interprocess_exception& e) {
        HCCL_VM_ERROR("[IsShmExists] 错误：打开共享内存失败 - {}", e.what());
        return false;
    }
}

void SHMManager::InitShm(bool create, size_t size)
{
    if (s_initialized.load(std::memory_order_acquire)) {
        return;
    }

    try {
        if (create) {
            ipc::shared_memory_object::remove(SHM_NAME);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            ipc::permissions perm;
            perm.set_unrestricted();

            s_shm = ipc::managed_shared_memory(ipc::create_only, SHM_NAME, size, 0, perm);
            s_shm.construct<ipc::interprocess_mutex>(SHM_MUTEX_NAME)();
            HCCL_VM_INFO("[SHMManager] 成功创建共享内存：{}，大小：{:d} 字节", SHM_NAME, size);
        } else {
            // 尝试打开共享内存
            s_shm = ipc::managed_shared_memory(ipc::open_only, SHM_NAME);
            // 验证是否真的打开成功（检查段管理器）
            if (!s_shm.get_segment_manager()) {
                throw std::runtime_error("共享内存打开失败：段管理器为空");
            }
            HCCL_VM_INFO("[SHMManager] 成功打开共享内存：{}", SHM_NAME);
        }

        s_initialized.store(true, std::memory_order_release);
    } catch (const ipc::interprocess_exception &e) {
        std::string err_msg = "[SHMManager] 初始化失败：";
        // 改用原生错误码判断（兼容所有 Boost 旧版本）
        int native_err = e.get_native_error();
        if (native_err == ENOENT) {  // 对应 "文件不存在"（共享内存未创建）
            err_msg += "共享内存不存在（请先启动主进程）";
        } else if (native_err == EACCES || native_err == EPERM) {  // 对应 "权限不足"
            err_msg += "权限不足（请以root运行）";
        } else if (native_err == EEXIST) {  // 对应 "文件已存在"（共享内存已创建）
            err_msg += "共享内存已存在（清理残留后重试）";
        } else {
            err_msg += "未知错误";
            err_msg += e.what();
        }
        HCCL_VM_ERROR("{}", err_msg);
        throw std::runtime_error(err_msg);
    }

    // 最终兜底：如果走到这里但 s_initialized 仍为 false，强制抛异常
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] 初始化失败：共享内存未打开");
        throw std::runtime_error("[SHMManager] 初始化失败：共享内存未打开");
    }
}

void SHMManager::DestroyShm()
{
    ipc::shared_memory_object::remove(SHM_NAME);
    s_initialized.store(false, std::memory_order_release);
    HCCL_VM_INFO("[SHMManager] 共享内存销毁完成：{}", SHM_NAME);
}

void *SHMManager::AllocateShmMemory(uint64_t size, const std::string &memkey)
{
    if (size == 0) {
        throw std::invalid_argument("SHMManager: 分配大小不能为0");
    }

    auto *mutex = FindShmObject<ipc::interprocess_mutex>(SHM_MUTEX_NAME);
    if (!mutex) {
        throw std::runtime_error("SHMManager: 未找到HcclSimSharedMemoryMutex");
    }

    ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);

    // 步骤1：检查占位符是否已存在（避免重复创建）
    if (FindShmObject<ShmBufferPlaceholder>(memkey) != nullptr) {
        throw std::runtime_error("SHMManager: 内存块'" + memkey + "'已存在");
    }

    // 步骤2：分配实际缓冲区内存（原始内存，大小为用户指定的size）
    void *buffer_ptr = s_shm.allocate(size);
    if (!buffer_ptr) {
        HCCL_VM_ERROR("[SHMManager] 分配共享内存失败：内存不足");
        throw std::bad_alloc();
    }
    std::memset(buffer_ptr, 0, size);  // 初始化内存

    // 步骤3：创建占位符对象，存储实际缓冲区地址（用于后续查找）
    ShmBufferPlaceholder *placeholder = s_shm.construct<ShmBufferPlaceholder>(memkey.c_str())();
    placeholder->buffer_ptr = buffer_ptr;  // 关联地址

    return buffer_ptr;  // 返回实际缓冲区地址，用户层无感知
}

ShmSegmentManager *SHMManager::GetSegmentManager()
{
    if (!s_shm.get_segment_manager()) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化");
        throw std::runtime_error("SHMManager: 共享内存未初始化");
    }
    return s_shm.get_segment_manager();
}

ipc::managed_shared_memory& SHMManager::GetSegment() {
    if (!s_shm.get_segment_manager()) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化");
        throw std::runtime_error("SHMManager: 共享内存未初始化");
    }
    return s_shm;
}

// -------------------------- SHMManager 业务模块接口实现 --------------------------
// 1. CheckOpParam 模块
CheckOpParam *SHMManager::CreateCheckOpParamModule()
{
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化，请先调用 InitShm");
        throw std::runtime_error("SHMManager: 共享内存未初始化，请先调用 InitShm");
    }

    return ConstructShmObject<CheckOpParam>(SHM_MODULE_CHECK_OP_PARAM, GetSegmentManager());
}

CheckOpParam *SHMManager::GetCheckOpParamModule()
{
    return FindShmObject<CheckOpParam>(SHM_MODULE_CHECK_OP_PARAM);
}

ProxyConfig* SHMManager::CreateProxyConfig()
{
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化，请先调用 InitShm");
        throw std::runtime_error("SHMManager: 共享内存未初始化，请先调用 InitShm");
    }

    return ConstructShmObject<ProxyConfig>(SHM_MODULE_PROXY_CONFIG);
}

ProxyConfig* SHMManager::GetProxyConfig()
{
    return FindShmObject<ProxyConfig>(SHM_MODULE_PROXY_CONFIG);
}

// 2. SimWorld 模块
ShmSimWorld *SHMManager::CreateSimWorldModule(const TopoMeta& topoMeta)
{
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化，请先调用 InitShm");
        throw std::runtime_error("SHMManager: 共享内存未初始化，请先调用 InitShm");
    }

    uint32_t deviceNum = ShmGetPhyDeviceTotalCount(topoMeta);
    if (deviceNum == 0 || deviceNum > MAX_DEV_NUM) {
        HCCL_VM_ERROR("CreateSimWorldModule: dev_size 无效（0 < dev_size <= {:d}）", MAX_DEV_NUM);
        throw std::invalid_argument(
            "CreateSimWorldModule: dev_size 无效（0 < dev_size <= " + std::to_string(MAX_DEV_NUM) + "）");
    }

    // 创建SimWorld
    auto *sim_world = ConstructShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD);
    sim_world->devNum = deviceNum;

    // 创建NpuPos2IndexMap
    ShmMapAllocator<uint32_t, size_t> allocator(s_shm.get_segment_manager());
    ConstructShmObject<NpuPos2Index>(SHM_MODULE_NPU_IDX_MAP, allocator);

    // 创建MockProxyBuffer
    ConstructShmObject<MockProxyBuffer>(SHM_MODULE_MOCK_PROXY_BUFFER);
    ConstructShmObject<uint32_t>(SHM_MODULE_MODE, s_mode);

    // 创建CommDomain
    ConstructShmObject<ShmCommDomain>(SHM_MODULE_COMM_DOMAIN);

    auto ret = InitSimWorld(&topoMeta);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Init SimWorld Failed!");
        throw std::runtime_error("Init SimWorld Failed!");
    }

    return sim_world;
}

ShmSimWorld *SHMManager::GetSimWorldModule()
{
    return FindShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD);
}

// 3. ProxyBuffer 模块
ProxyBuffer* SHMManager::CreateProxyBufferModule(size_t bufferSize)
{
    // zhf-创建MockProxyBuffer
    ConstructShmObject<MockProxyBuffer>(SHM_MODULE_MOCK_PROXY_BUFFER);
    ConstructShmObject<uint32_t>(SHM_MODULE_MODE, s_mode);

    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化，请先调用 InitShm");
        throw std::runtime_error("SHMManager: 共享内存未初始化，请先调用 InitShm");
    }
    if (bufferSize == 0) {
        HCCL_VM_ERROR("CreateProxyBufferModule: buffer_size 不能为0");
        throw std::invalid_argument("CreateProxyBufferModule: buffer_size 不能为0");
    }

    // 创建ProxyBuffer
    auto* proxyBuffer = ConstructShmObject<ProxyBuffer>(SHM_MODULE_PROXY_BUFFER);
    // ProxyBuffer初始化
    ipc::scoped_lock<ipc::interprocess_mutex> lock(proxyBuffer->mutex);
    void* bufferPtr = s_shm.allocate(bufferSize);
    if (!bufferPtr) {
        HCCL_VM_ERROR("ProxyBuffer内存分配失败");
        throw std::bad_alloc();
    }
    proxyBuffer->totalSize = bufferSize;
    proxyBuffer->allocatedSize = 0;
    proxyBuffer->bufferPtr = bufferPtr;

    return proxyBuffer;
}

ProxyBuffer* SHMManager::GetProxyBufferModule()
{
    return FindShmObject<ProxyBuffer>(SHM_MODULE_PROXY_BUFFER);
}

// 4. TaskCollection 模块
TaskCollection *SHMManager::CreateTaskCollectionModule(uint32_t max_task_count)
{
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化，请先调用 InitShm");
        throw std::runtime_error("SHMManager: 共享内存未初始化，请先调用 InitShm");
    }
    if (max_task_count == 0 || max_task_count > SHM_TASK_COLLECTION_LENGTH) {
        HCCL_VM_ERROR("CreateTaskCollectionModule: max_task_count 无效");
        throw std::invalid_argument("CreateTaskCollectionModule: max_task_count 无效");
    }

    // 直接复用 AllocateShmMemory 分配柔性数组内存（自动创建 ShmBufferPlaceholder 占位符）
    size_t total_size = sizeof(TaskCollection) + sizeof(HcclTaskMetaData) * max_task_count;
    void *raw_mem = AllocateShmMemory(total_size, SHM_MODULE_TASK_COLLECTION);

    // placement-new 构造对象（无需手动锁，AllocateShmMemory 已加锁）
    TaskCollection *task_collection = new (raw_mem) TaskCollection(max_task_count);

    // 创建专属互斥锁（逻辑不变）
    std::string mutex_name = std::string(SHM_MODULE_TASK_COLLECTION) + "_Mutex";
    auto *task_mutex = s_shm.construct<ipc::interprocess_mutex>(mutex_name.c_str())();
    if (!task_mutex) {
        // 回滚：析构对象 + 调用现有 Destroy 逻辑（自动释放内存+占位符）
        task_collection->~TaskCollection();
        DestroyShmObject<void>(SHM_MODULE_TASK_COLLECTION);
        HCCL_VM_ERROR("创建 {} 失败", mutex_name);
        throw std::runtime_error("SHMManager: 创建 " + mutex_name + " 失败");
    }

    HCCL_VM_INFO("✅ 成功创建 TaskCollection 模块（含互斥锁），最大任务数：{:d}", max_task_count);
    return task_collection;
}

TaskCollection *SHMManager::GetTaskCollectionModule()
{
    ShmBufferPlaceholder *placeholder = FindShmObject<ShmBufferPlaceholder>(SHM_MODULE_TASK_COLLECTION);
    if (!placeholder || !placeholder->buffer_ptr) {
        HCCL_VM_ERROR("[SHMManager] 未找到 TaskCollection 模块");
        throw std::runtime_error("SHMManager: 未找到 TaskCollection 模块");
    }
    // 通过offset_ptr得到当前进程中的绝对地址，然后强转为 TaskCollection*
    return static_cast<TaskCollection*>(placeholder->buffer_ptr.get());
}

bool SHMManager::AddTaskToCollection(TaskCollection *task_collect, const HcclTaskMetaData &task)
{
    if (!task_collect) {
        HCCL_VM_ERROR("AddTaskToCollection: task_collect 为空指针");
        throw std::invalid_argument("AddTaskToCollection: task_collect 为空指针");
    }

    std::string mutex_name = std::string(SHM_MODULE_TASK_COLLECTION) + "_Mutex";
    auto *mutex = FindShmObject<ipc::interprocess_mutex>(mutex_name);
    if (!mutex) {
        HCCL_VM_ERROR("[SHMManager] 未找到 TaskCollection 互斥锁");
        throw std::runtime_error("SHMManager: 未找到 TaskCollection 互斥锁");
    }

    ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);
    if (task_collect->indexGen >= task_collect->maxSize) {
        HCCL_VM_ERROR("[SHMManager] AddTaskToCollection: 任务数超出最大值（{:d}）", task_collect->maxSize);
        return false;
    }
    HCCL_VM_DEBUG("[{}] INSERT {:d}, {:d}, {:d}", __func__, task.rankId, task.streamId, (int)(task.taskType));
    task_collect->tasks[task_collect->indexGen] = task;
    task_collect->indexGen++;
    return true;
}

uint32_t SHMManager::GetTaskCollectionCount(TaskCollection *task_collect)
{
    if (!task_collect) {
        HCCL_VM_ERROR("GetTaskCollectionCount: task_collect 为空指针");
        throw std::invalid_argument("GetTaskCollectionCount: task_collect 为空指针");
    }

    std::string mutex_name = std::string(SHM_MODULE_TASK_COLLECTION) + "_Mutex";
    auto *mutex = FindShmObject<ipc::interprocess_mutex>(mutex_name);
    if (!mutex) {
        HCCL_VM_ERROR("[SHMManager] 未找到 TaskCollection 互斥锁");
        throw std::runtime_error("SHMManager: 未找到 TaskCollection 互斥锁");
    }

    ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);
    return task_collect->indexGen;
}

bool SHMManager::GetTaskFromCollection(TaskCollection *task_collect, uint32_t idx, HcclTaskMetaData &out_task)
{
    if (!task_collect) {
        HCCL_VM_ERROR("GetTaskFromCollection: task_collect 为空指针");
        throw std::invalid_argument("GetTaskFromCollection: task_collect 为空指针");
    }

    std::string mutex_name = std::string(SHM_MODULE_TASK_COLLECTION) + "_Mutex";
    auto *mutex = FindShmObject<ipc::interprocess_mutex>(mutex_name);
    if (!mutex) {
        HCCL_VM_ERROR("[SHMManager] 未找到 TaskCollection 互斥锁");
        throw std::runtime_error("SHMManager: 未找到 TaskCollection 互斥锁");
    }

    ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);
    if (idx >= task_collect->indexGen) {
        HCCL_VM_ERROR("[SHMManager] GetTaskFromCollection: 索引 {:d} 超出任务总数 {:d}", idx, task_collect->indexGen);
        return false;
    }

    out_task = task_collect->tasks[idx];
    return true;
}

// 5. ProxyOffsetArray 模块
ProxyOffsetArray *SHMManager::CreateProxyOffsetArrayModule(size_t init_offset_count)
{
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] 共享内存未初始化，请先调用 InitShm");
        throw std::runtime_error("SHMManager: 共享内存未初始化，请先调用 InitShm");
    }
    return ConstructShmObject<ProxyOffsetArray>(SHM_MODULE_PROXY_OFFSET_ARRAY, GetSegmentManager(), init_offset_count);
}

ProxyOffsetArray *SHMManager::GetProxyOffsetArrayModule()
{
    return FindShmObject<ProxyOffsetArray>(SHM_MODULE_PROXY_OFFSET_ARRAY);
}

bool SHMManager::CreateSingleMessageQueue(const std::string &queue_name, size_t max_msg_count, size_t msg_size)
{
    try {
        ShmSegmentManager *seg_mgr = GetSegmentManager();
        if (!seg_mgr) {
            HCCL_VM_ERROR("[SHMManager] 创建队列失败：共享内存段管理器为空 - {}", queue_name);
            return false;
        }

        // 先删除共享内存中已存在的同名队列
        DestroySingleMessageQueue(queue_name);

        // 创建队列（绑定到 HcclSimSharedMemory 共享内存）
        ipc::message_queue mq(ipc::create_only, queue_name.c_str(), max_msg_count, msg_size, ipc::permissions(0666));
        HCCL_VM_INFO("[SHMManager] 队列创建成功：{}", queue_name);
        HCCL_VM_INFO("  - 最大消息数：{:d}", max_msg_count);
        HCCL_VM_INFO("  - 单条消息大小：{:d} 字节", msg_size);
        return true;
    } catch (const ipc::interprocess_exception &e) {
        HCCL_VM_ERROR("[SHMManager] 创建队列失败：{} - {}", queue_name, e.what());
        return false;
    }
}

bool SHMManager::DestroySingleMessageQueue(const std::string &queue_name)
{
    try {
        // 仅删除共享内存中的队列（不影响系统IPC）
        ipc::message_queue::remove(queue_name.c_str());
        return true;
    } catch (const ipc::interprocess_exception &e) {
        if (e.get_native_error() == ENOENT) {
            return true;
        }
        HCCL_VM_ERROR("[SHMManager] 删除队列失败：{} - {}", queue_name, e.what());
        return false;
    }
}

// -------------------------- 新增 IPC 对外接口实现（public） --------------------------
HcclVmResult SHMManager::InitIpc(uint32_t rankNum, size_t task_req_size, size_t task_rsp_size)
{
    // 1. 入参合法性检查
    if (rankNum == 0) {
        HCCL_VM_ERROR("[SHMManager] InitIpc 错误：rankNum 不能为 0！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (task_req_size == 0 || task_rsp_size == 0) {
        HCCL_VM_ERROR("[SHMManager] InitIpc 错误：消息大小不能为 0！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] InitIpc 错误：共享内存未初始化！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    try {
        const size_t MAX_MSG_COUNT = 10240;  // 队列最大消息数（可配置为常量）

        // 2. 创建固定队列：runtime_0（HcclTaskReq 类型）
        if (!CreateSingleMessageQueue("runtime_0", MAX_MSG_COUNT, task_req_size)) {
            throw std::runtime_error("创建 runtime_0 队列失败");
        }

        // 3. 创建动态队列：proxy_0 ~ proxy_{rankNum-1}（HcclTaskRsp 类型）
        for (uint32_t i = 0; i < rankNum; ++i) {
            std::string queue_name = "proxy_" + std::to_string(i);
            if (!CreateSingleMessageQueue(queue_name, MAX_MSG_COUNT, task_rsp_size)) {
                // 部分队列创建失败，回滚已创建的队列
                DestroyIpc(i);                           // 删除已创建的 proxy_0 ~ proxy_{i-1}
                DestroySingleMessageQueue("runtime_0");  // 删除 runtime_0
                throw std::runtime_error("创建 " + queue_name + " 队列失败");
            }
        }

        HCCL_VM_INFO("[SHMManager] InitIpc 成功：共创建 {:d} 个队列", (1 + rankNum));
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("[SHMManager] InitIpc 失败：{}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
}

HcclVmResult SHMManager::DestroyIpc(uint32_t rankNum)
{
    if (!s_initialized.load(std::memory_order_acquire)) {
        HCCL_VM_ERROR("[SHMManager] DestroyIpc 错误：共享内存未初始化！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 1. 删除动态队列：proxy_0 ~ proxy_{rankNum-1}
    for (uint32_t i = 0; i < rankNum; ++i) {
        std::string queue_name = "proxy_" + std::to_string(i);
        DestroySingleMessageQueue(queue_name);
    }

    // 2. 删除固定队列：runtime_0
    DestroySingleMessageQueue("runtime_0");

    HCCL_VM_INFO("[SHMManager] DestroyIpc 成功：共删除 {:d} 个队列", (1 + rankNum));
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

void SHMManager::SetHcclVmMode(uint32_t mode) {
    SHMManager::InitShm(false);
    uint32_t* shmMode = SHMManager::FindShmObject<uint32_t>(SHM_MODULE_MODE);
    if (shmMode == nullptr) {
        HCCL_VM_WARN("[SHMManager] [SetHcclVmMode] 未找到SHM_MODULE_MODE");
    }
    *shmMode = mode;
    return;
}

uint32_t SHMManager::GetHcclVmMode() {
    return 0;
    // todo: 获取mockMode有问题，待定位
    // uint32_t* mockMode = SHMManager::FindShmObject<uint32_t>(SHM_MODULE_MODE);
    // return *mockMode;
}

// 批量销毁所有模块
void SHMManager::DestroyAllModules()
{
    if (!s_shm.get_segment_manager()) {
        HCCL_VM_WARN("[SHMManager] 共享内存未初始化，无需销毁模块");
        return;
    }

    // 销毁 TaskCollection 专属互斥锁（先判断是否存在）
    std::string task_mutex_name = std::string(SHM_MODULE_TASK_COLLECTION) + "_Mutex";
    if (FindShmObject<ipc::interprocess_mutex>(task_mutex_name)) {
        s_shm.destroy<ipc::interprocess_mutex>(task_mutex_name.c_str());
    }

    // 销毁其他模块（均判断存在性）
    if (FindShmObject<ProxyOffsetArray>(SHM_MODULE_PROXY_OFFSET_ARRAY)) {
        DestroyShmObject<ProxyOffsetArray>(SHM_MODULE_PROXY_OFFSET_ARRAY);
    }
    if (FindShmObject<TaskCollection>(SHM_MODULE_TASK_COLLECTION)) {
        DestroyShmObject<TaskCollection>(SHM_MODULE_TASK_COLLECTION);
    }
    ProxyBuffer* proxyBuffer = FindShmObject<ProxyBuffer>(SHM_MODULE_PROXY_BUFFER);
    if (proxyBuffer) {
        if (proxyBuffer->bufferPtr) {
            s_shm.deallocate(proxyBuffer->bufferPtr.get());
        }
        DestroyShmObject<ProxyBuffer>(SHM_MODULE_PROXY_BUFFER);
    }
    if (FindShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD)) {
        DestroyShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD);
    }
    if (FindShmObject<CheckOpParam>(SHM_MODULE_CHECK_OP_PARAM)) {
        DestroyShmObject<CheckOpParam>(SHM_MODULE_CHECK_OP_PARAM);
    }
    HCCL_VM_INFO("[SHMManager] 所有模块销毁完成");
}

void SHMManager::ResetGlobalState()
{
    s_initialized.store(false, std::memory_order_seq_cst);
}