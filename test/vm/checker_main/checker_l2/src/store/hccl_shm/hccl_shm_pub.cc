#include <iostream>
#include <mutex>

#include <store/sim_shm_memory_manager.h>
#include <store/hccl_shm_pub.h>
#include <store/hccl_sim_shm_manager.h>
#include <store/sim_shm_vir_memory_manager.h>
#include "hccl_vm_log.h"

using namespace HcclSim;

// Virtual Runtime
HcclVmResult GetAddrByOffset(uint64_t offset, void **addr)  // todo
{
    // 1. 入参合法性检查（避免空指针和无效输出）
    if (addr == nullptr) {
        HCCL_VM_ERROR("[GetAddrByOffset] 错误：输出参数 addr 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 获取 ProxyBuffer 模块基地址（共享内存中的起始地址）
    SHMManager::InitShm(false);
    ProxyBuffer* proxyBuffer = SHMManager::GetProxyBufferModule();
    if (proxyBuffer == nullptr) {
        HCCL_VM_ERROR("[GetAddrByOffset] 错误：未找到 ProxyBuffer 模块（请先调用 InitSharedMemory 初始化）！");
        return HcclVmResult::HCCL_SIM_E_PTR;
    }

    // 3. 偏移量合法性检查（避免超出 ProxyBuffer 最大容量，防止内存越界）
    if (offset > proxyBuffer->totalSize) {
        HCCL_VM_ERROR("[GetAddrByOffset] 错误：偏移量 {:d} 超出 ProxyBuffer 最大容量 {:d} 字节！", offset, proxyBuffer->totalSize);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 4. 计算最终地址：基地址 + 偏移量（安全类型转换）
    *addr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(proxyBuffer->bufferPtr.get()) + offset);

    // 5. 调试日志（可选，便于问题排查）
    HCCL_VM_INFO("[GetAddrByOffset] 计算成功：");
    HCCL_VM_INFO("  - ProxyBuffer 基地址：{:p}", proxyBuffer->bufferPtr.get());
    HCCL_VM_INFO("  - 输入偏移量：{:d} 字节", offset);
    HCCL_VM_INFO("  - 输出地址：{:p}", *addr);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetTaskCollectionByCid(uint64_t taskCid, HcclTaskMetaData *task)
{
    // 1. 入参合法性检查（避免空指针访问）
    if (task == nullptr) {
        HCCL_VM_ERROR("[GetTaskCollectionByCid] 错误：输出参数 task 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PTR;
    }

    // 2. 解析 TaskCid 中的 index 字段（联合体自动解析，无需手动位运算）
    HcclTaskCid cid;
    cid.value = taskCid;  // 将输入的 uint64_t 赋值给联合体，自动映射到 field 成员
    uint32_t target_index = cid.field.index;

    // 3. 获取 TaskCollection 模块（检查是否已初始化）
    SHMManager::InitShm(false);
    TaskCollection *task_collection = SHMManager::GetTaskCollectionModule();
    if (task_collection == nullptr) {
        HCCL_VM_ERROR("[GetTaskCollectionByCid] 错误：未找到 TaskCollection 模块（请先调用 InitSharedMemory 初始化）！");
        return HcclVmResult::HCCL_SIM_SHM_TASK_COLLECTION_ERROR;
    }

    // 4. 检查 index 有效性（避免越界访问）
    uint32_t task_count = SHMManager::GetTaskCollectionCount(task_collection);
    if (target_index >= task_count) {
        HCCL_VM_ERROR("[GetTaskCollectionByCid] 错误：索引 {:d} 超出有效范围！", target_index);
        HCCL_VM_ERROR("  - 已存储任务数：{:d}", task_count);
        HCCL_VM_ERROR("  - 输入 TaskCid：{:x}", taskCid);
        HCCL_VM_ERROR("  - 解析字段：commId={:d}, rankId={:d}, index={:d}",
            static_cast<uint64_t>(cid.field.commId), static_cast<uint64_t>(cid.field.rankId), target_index);
        return HcclVmResult::HCCL_SIM_SHM_TASK_COLLECTION_ERROR;
    }

    // 5. 从 TaskCollection 中读取对应索引的任务元数据
    bool get_success = SHMManager::GetTaskFromCollection(task_collection, target_index, *task);
    if (!get_success) {
        HCCL_VM_ERROR("[GetTaskCollectionByCid] 错误：读取索引 {:d} 的任务失败！", target_index);
        return HcclVmResult::HCCL_SIM_SHM_TASK_COLLECTION_ERROR;
    }

    // 6. 调试日志（可选，便于问题排查）
    HCCL_VM_INFO("[GetTaskCollectionByCid] 读取任务成功：");
    HCCL_VM_INFO("  - TaskCid（十六进制）：{:x}", taskCid);
    HCCL_VM_INFO("  - 解析字段：commId={:d}, rankId={:d}, index={:d}",
        static_cast<uint64_t>(cid.field.commId), static_cast<uint64_t>(cid.field.rankId), target_index);
    HCCL_VM_INFO("  - 任务索引：{:d}（总任务数：{:d}）", target_index, task_count);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AllocRankMemForProxy(uint64_t size, void **memAddr)    // todo
{
    // 1. 入参合法性检查
    if (memAddr == nullptr) {
        HCCL_VM_ERROR("[AllocRankMemForProxy] 错误：输出参数 memAddr 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (size == 0) {
        HCCL_VM_ERROR("[AllocRankMemForProxy] 错误：分配内存大小不能为0！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 ProxyBuffer 模块是否已初始化
    SHMManager::InitShm(false);
    ProxyBuffer* proxyBuffer = SHMManager::GetProxyBufferModule();
    if (proxyBuffer == nullptr) {
        HCCL_VM_ERROR("[AllocRankMemForProxy] 错误：ProxyBuffer 模块未初始化（请先调用 InitSharedMemory）！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 3. 核心分配逻辑（加锁保证进程安全）
    ipc::scoped_lock<ipc::interprocess_mutex> lock(proxyBuffer->mutex);

    // 检查内存是否充足（已分配偏移 + 申请大小 <= ProxyBuffer 总容量）
    if (proxyBuffer->allocatedSize + size > proxyBuffer->totalSize) {
        HCCL_VM_ERROR("[AllocRankMemForProxy] 错误：ProxyBuffer 内存不足！");
        HCCL_VM_ERROR("  - 已分配：{:d} 字节", proxyBuffer->allocatedSize);
        HCCL_VM_ERROR("  - 申请：{:d} 字节", size);
        HCCL_VM_ERROR("  - 总容量：{:d} 字节", proxyBuffer->totalSize);
        return HcclVmResult::HCCL_SIM_SHM_BUFFER_OUT_OF_MEMORY;
    }

    // 计算分配内存的首地址（基地址 + 已分配偏移）
    *memAddr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(proxyBuffer->bufferPtr.get()) + proxyBuffer->allocatedSize);
    // 更新已分配偏移（供下一次分配使用）
    proxyBuffer->allocatedSize += size;
    // 调试日志
    HCCL_VM_INFO("[AllocRankMemForProxy] 内存分配成功：");
    HCCL_VM_INFO("  - 分配大小：{:d} 字节", size);
    HCCL_VM_INFO("  - ProxyBuffer 基地址：{:p}", proxyBuffer->bufferPtr.get());
    HCCL_VM_INFO("  - 分配内存首地址：{:p}", *memAddr);
    HCCL_VM_INFO("  - 已分配总大小：{:d} 字节", proxyBuffer->allocatedSize);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetProxyBufferMemOffset(const void* addr, uint64_t* offset)
{
    // 1. 入参合法性检查（避免空指针和无效输入）
    if (offset == nullptr) {
        HCCL_VM_ERROR("[GetProxyBufferMemOffset] 错误：输出参数 offset 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (addr == nullptr) {
        HCCL_VM_ERROR("[GetProxyBufferMemOffset] 错误：输入地址 addr 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 ProxyBuffer 模块是否已初始化
    SHMManager::InitShm(false);
    ProxyBuffer* proxyBuffer = SHMManager::GetProxyBufferModule();
    if (proxyBuffer == nullptr) {
        HCCL_VM_ERROR("[GetProxyBufferMemOffset] 错误：ProxyBuffer 模块未初始化（请先调用 InitSharedMemory）！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 3. 获取基准偏移量
    uint64_t baseOffset = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(proxyBuffer->bufferPtr.get()));
    HCCL_VM_INFO("[GetProxyBufferMemOffset] ProxyBuffer Base地址：{:p}", proxyBuffer->bufferPtr.get());

    // 4. 地址转换与偏移量计算（按字节计算，避免类型转换错误）
    uint64_t addrValue = reinterpret_cast<uint64_t>(static_cast<const uint8_t*>(addr));  // 地址转为数值

    // 5. 校验偏移量合法性（地址不能小于基准偏移量）
    if (addrValue < baseOffset) {
        HCCL_VM_ERROR("[GetProxyBufferMemOffset] 错误：输入地址 {:p} 小于 ProxyBuffer 的基准偏移量 {:x}！", addr, baseOffset);
        return HcclVmResult::HCCL_SIM_SHM_PROXY_OUT_OF_RANGE;
    }

    // 6. 计算最终偏移量：输入地址 - 基准偏移量，输出结果
    *offset = addrValue - baseOffset;
    HCCL_VM_INFO("[GetProxyBufferMemOffset] 计算成功：");
    HCCL_VM_INFO("  - 输入地址：{:p}", addr);
    HCCL_VM_INFO("  - Base地址：{:p}", proxyBuffer->bufferPtr.get());
    HCCL_VM_INFO("  - 输出偏移量：{:d} 字节", *offset);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetMockBufferMemOffset(const void* addr, uint64_t* offset) {
    // 1. 入参合法性检查（避免空指针和无效输入）
    if (offset == nullptr) {
        HCCL_VM_ERROR("[GetMockBufferMemOffset] 错误：输出参数 offset 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (addr == nullptr) {
        HCCL_VM_ERROR("[GetMockBufferMemOffset] 错误：输入地址 addr 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 ProxyBuffer 模块是否已初始化
    SHMManager::InitShm(false);
    ProxyBuffer* proxyBuffer = SHMManager::GetProxyBufferModule();
    if (proxyBuffer == nullptr) {
        HCCL_VM_ERROR("[GetMockBufferMemOffset] GetProxyBufferModule 错误: ProxyBuffer 模块未初始化（请先调用 InitSharedMemory）");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    MockProxyBuffer* mockBuffer = SHMManager::FindShmObject<MockProxyBuffer>(SHM_MODULE_MOCK_PROXY_BUFFER);
    uint64_t baseValue = reinterpret_cast<uint64_t>(mockBuffer->mockBase);
    uint64_t addrValue = reinterpret_cast<uint64_t>(addr);
    if (addrValue < baseValue) {
        HCCL_VM_ERROR("[GetMockBufferMemOffset] 错误：输入地址 {:p} 小于 MockBufferBase {:x}", addr, baseValue);
        return HcclVmResult::HCCL_SIM_SHM_PROXY_OUT_OF_RANGE;
    }

    *offset = addrValue - baseValue;
    HCCL_VM_INFO("[GetMockBufferMemOffset] 计算成功：");
    HCCL_VM_INFO("  - 输入地址：{:x}", addrValue);
    HCCL_VM_INFO("  - Base地址：{:x}", baseValue);
    HCCL_VM_INFO("  - 输出偏移量：{:d} 字节", *offset);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetRankOffset(uint32_t rankId, const void *addr, uint64_t *offset)
{
    // 1. 入参合法性检查（避免空指针和无效输入）
    if (offset == nullptr) {
        HCCL_VM_ERROR("[GetRankOffset] 错误：输出参数 offset 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (addr == nullptr) {
        HCCL_VM_ERROR("[GetRankOffset] 错误：输入地址 addr 不能为空指针！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (rankId >= MAX_DEV_NUM) {  // 复用 MAX_DEV_NUM 限制 rankId 范围（避免下标越界）
        HCCL_VM_ERROR("[GetRankOffset] 错误：rankId {:d} 超出最大限制 {:d}！", rankId, MAX_DEV_NUM);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 ProxyOffsetArray 模块是否已初始化
    SHMManager::InitShm(false);
    ProxyOffsetArray *offset_array = SHMManager::GetProxyOffsetArrayModule();
    if (offset_array == nullptr) {
        HCCL_VM_ERROR("[GetRankOffset] 错误：ProxyOffsetArray 模块未初始化（请先调用 InitSharedMemory）！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 3. 核心逻辑：读取 rankId 对应的基准偏移量
    try {
        // 从 ProxyOffsetArray 中读取 rankId 下标对应的基准偏移量（默认值 0）
        uint64_t base_offset = offset_array->ReadOffset(rankId, 0);
        HCCL_VM_INFO("[GetRankOffset] 读取 rankId={:d} 的基准偏移量：{:x}", rankId, base_offset);

        // 4. 地址转换与偏移量计算（按字节计算，避免类型转换错误）
        const uint8_t *addr_byte = reinterpret_cast<const uint8_t *>(addr);  // 转为字节指针
        uint64_t addr_value = reinterpret_cast<uint64_t>(addr_byte);  // 地址转为数值（兼容 64 位系统）

        // 计算最终偏移量：输入地址 - 基准偏移量
        int64_t calculated_offset = static_cast<int64_t>(addr_value - base_offset);

        // 5. 校验偏移量合法性（不能为负，避免地址小于基准偏移量）
        if (calculated_offset < 0) {
            HCCL_VM_ERROR("[GetRankOffset] 错误：输入地址 {:p} 小于 rankId={:d} 的基准偏移量 {:x}！", addr, rankId, base_offset);
            HCCL_VM_ERROR("  - 计算结果：{:x}（非法负偏移）", calculated_offset);
            return HcclVmResult::HCCL_SIM_SHM_PROXY_OUT_OF_RANGE;
        }

        // 6. 输出结果
        *offset = static_cast<uint64_t>(calculated_offset);
        HCCL_VM_INFO("[GetRankOffset] 计算成功：");
        HCCL_VM_INFO("  - 输入地址：{:p}", addr);
        HCCL_VM_INFO("  - 基准偏移量（rankId={:d}）：{:x}", rankId, base_offset);
        HCCL_VM_INFO("  - 输出偏移量：{:d} 字节", *offset);

        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("[GetRankOffset] 错误：计算偏移量失败 - {}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
}

HcclVmResult InsertTaskToCollection(HcclTaskMetaData *task, uint32_t *index)
{
    // 1. 入参合法性检查（避免空指针访问）
    if (task == nullptr) {
        HCCL_VM_ERROR("[InsertTaskToCollection] 错误：输入任务指针 task 不能为空！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (index == nullptr) {
        HCCL_VM_ERROR("[InsertTaskToCollection] 错误：输出索引指针 index 不能为空！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 TaskCollection 模块是否已初始化
    SHMManager::InitShm(false);
    TaskCollection *task_collection = SHMManager::GetTaskCollectionModule();
    if (task_collection == nullptr) {
        HCCL_VM_ERROR("[InsertTaskToCollection] 错误：TaskCollection 模块未初始化（请先调用 InitSharedMemory）！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 3. 核心插入逻辑（线程安全，依赖 TaskCollection 专属互斥锁）
    try {
        std::string mutex_name = std::string(SHM_MODULE_TASK_COLLECTION) + "_Mutex";
        ipc::interprocess_mutex *mutex = SHMManager::FindShmObject<ipc::interprocess_mutex>(mutex_name);
        if (!mutex) {
            throw std::runtime_error("未找到 TaskCollection 互斥锁");
        }

        // 加锁保护插入操作（避免多线程并发插入冲突）
        ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);

        // 4. 检查 TaskCollection 是否已满
        if (task_collection->indexGen >= task_collection->maxSize) {
            HCCL_VM_ERROR("[InsertTaskToCollection] 错误：TaskCollection 已满！");
            HCCL_VM_ERROR("  - 最大任务数：{:d}", task_collection->maxSize);
            HCCL_VM_ERROR("  - 当前任务数：{:d}", task_collection->indexGen);
            return HcclVmResult::HCCL_SIM_SHM_TASK_COLLECTION_ERROR;
        }

        // 5. 插入任务并记录索引
        uint32_t task_index = task_collection->indexGen;  // 当前索引（插入前的 indexGen）
        task_collection->tasks[task_index] = *task;       // 拷贝任务数据到共享内存
        task_collection->indexGen++;                      // 索引生成器自增（下一个任务的索引）

        // 6. 输出结果（返回插入后的任务下标）
        *index = task_index;
        HCCL_VM_INFO("[InsertTaskToCollection] 任务插入成功：");
        HCCL_VM_INFO("  - 插入索引：{:d}", *index);
        HCCL_VM_INFO("  - 当前任务总数：{:d}", task_collection->indexGen);
        HCCL_VM_INFO("  - 任务 rankId（示例）：{:d}", task->rankId);  // 修复：taskId → rankId

        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("[InsertTaskToCollection] 错误：任务插入失败 - {}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
}

HcclVmResult InsertProxyBaseAddr(uint32_t rankId)
{
    // 1. 入参合法性检查（rankId 不能超出最大设备数）
    if (rankId >= MAX_DEV_NUM) {
        HCCL_VM_ERROR("[InsertProxyBaseAddr] 错误：rankId {:d} 超出最大限制 {:d}！", rankId, MAX_DEV_NUM);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查核心模块是否已初始化
    SHMManager::InitShm(false);
    ProxyOffsetArray *offset_array = SHMManager::GetProxyOffsetArrayModule();
    if (offset_array == nullptr) {
        HCCL_VM_ERROR("[InsertProxyBaseAddr] 错误：ProxyOffsetArray 模块未初始化（请先调用 InitSharedMemory）！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 3. 检查共享内存是否初始化（通过 ProxyBuffer 模块验证，变量定义在外部，确保作用域覆盖后续代码）
    void *proxy_buffer = SHMManager::GetProxyBufferModule();
    if (proxy_buffer == nullptr) {
        HCCL_VM_ERROR("[InsertProxyBaseAddr] 错误：共享内存未初始化（请先调用 InitShm）！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 4. 核心逻辑：获取共享内存基地址（ProxyBuffer 地址即为共享内存基地址）
    try {
        // 修复变量作用域：proxy_buffer 定义在 try 外部，此处可直接访问
        void *shm_base_addr = proxy_buffer;
        if (shm_base_addr == nullptr) {
            throw std::runtime_error("共享内存映射基地址获取失败（ProxyBuffer 模块地址为空）");
        }

        // 将基地址转换为 uint64_t 类型（便于存储和偏移量计算）
        uint64_t shm_base_offset = reinterpret_cast<uint64_t>(shm_base_addr);

        // 5. 写入 ProxyOffsetArray（以 rankId 为下标）
        offset_array->WriteOffset(rankId, shm_base_offset);

        // 6. 日志输出（便于排查）
        HCCL_VM_INFO("[InsertProxyBaseAddr] 写入成功：");
        HCCL_VM_INFO("  - rankId：{:d}", rankId);
        HCCL_VM_INFO("  - 当前进程共享内存映射基地址：{:p}", shm_base_addr);
        HCCL_VM_INFO("  - 存储的基地址（uint64_t）：{:x}", shm_base_offset);

        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("[InsertProxyBaseAddr] 错误：写入共享内存基地址失败 - {}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
}

HcclSim::HcclVmResult ShmEnvInit()
{
    sim::shm::ShmMemoryManager::Initialize(true);
    sim::shm::VirMemoryManager::GetInstance().Initialize(true);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// Host
HcclVmResult InitIpc(uint32_t rankNum, uint32_t serverNum)
{
    (void)serverNum;
    // 1. 先初始化共享内存（创建模式）
    try {
        SHMManager::InitShm(true);  // 调用共享内存初始化（底层已处理 Boost）
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("[InitIpc] 共享内存初始化失败：{}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    // 2. 调用 SHMManager 的 IPC 封装接口（传入消息大小，避免底层依赖 HcclTaskReq/Rsp）
    return SHMManager::InitIpc(rankNum, sizeof(HcclTaskReq), sizeof(HcclTaskRsp));
}

HcclVmResult InitSharedMemory(TopoMeta topoMeta)
{
    // SHMManager::SetHcclVmMode(1);
    // std::cout << "  - HcclVM 运行模式: " << mode << " (mode 0 : checker | mode 1 : runner)"<<std::endl;
    uint32_t deviceNum = ShmGetPhyDeviceTotalCount(topoMeta);

    // 1. 参数合法性检查（提前拦截无效输入）
    if (deviceNum == 0 || deviceNum > MAX_DEV_NUM) {
        throw std::invalid_argument("InitAllShmModules: 设备数量无效！必须满足 0 < deviceNum <= " +
                                    std::to_string(MAX_DEV_NUM) + "（当前输入：" + std::to_string(deviceNum) + "）");
    }

    HCCL_VM_INFO("===== 开始初始化所有共享内存模块 =====");
    HCCL_VM_INFO("配置参数：");
    HCCL_VM_INFO("  - 设备数量：{:d}", deviceNum);
    HCCL_VM_INFO("  - ProxyBuffer 大小：{:d} 字节", SHM_DEFAULT_BUFFER_SIZE);
    HCCL_VM_INFO("  - 最大任务数：{:d}", SHM_TASK_COLLECTION_LENGTH);
    HCCL_VM_INFO("  - 初始偏移量数组大小：{:d}", deviceNum);

    // 2. 创建新的共享内存
    SHMManager::InitShm(true, SHM_DEFAULT_SIZE);

    // 3. 逐个创建模块（按依赖关系顺序）
    try {
        SHMManager::CreateProxyConfig();

        // 3.1 创建 CheckOpParam 模块（无额外参数）
        // SHMManager::CreateCheckOpParamModule();
        // std::cout << "✅ 成功创建 CheckOpParam 模块" << std::endl;

        // 3.2 创建 ShmSimWorld 模块
        // SHMManager::CreateSimWorldModule(topoMeta);
        // std::cout << "✅ 成功创建 ShmSimWorld 模块（设备数：" << deviceNum << "）" << std::endl;

        // 3.3 创建 ProxyBuffer 模块（传入缓冲区大小）
        SHMManager::CreateProxyBufferModule(SHM_DEFAULT_BUFFER_SIZE);
        HCCL_VM_INFO("✅ 成功创建 ProxyBuffer 模块（大小：{} 字节）", SHM_DEFAULT_BUFFER_SIZE);

        // 3.4 创建 TaskCollection 模块（传入最大任务数）
        SHMManager::CreateTaskCollectionModule(SHM_TASK_COLLECTION_LENGTH);
        HCCL_VM_INFO("✅ 成功创建 TaskCollection 模块（最大任务数：{}）", SHM_TASK_COLLECTION_LENGTH);

        // 3.5 创建 ProxyOffsetArray 模块（传入初始偏移量数组大小）
        // SHMManager::CreateProxyOffsetArrayModule(deviceNum);
        // std::cout << "✅ 成功创建 ProxyOffsetArray 模块（初始大小：" << deviceNum << "）" << std::endl;

        HCCL_VM_INFO("===== 所有共享内存模块初始化完成 =====");
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (...) {
        // 异常处理：创建失败时销毁已创建的模块和共享内存，避免残留
        HCCL_VM_ERROR("\n❌ 模块创建失败, 开始清理共享内存...");
        SHMManager::DestroyShm();
        throw std::runtime_error("InitAllShmModules: 初始化失败");
    }
}

// Plugin
HcclVmResult GetTaskCollection(HcclTaskMetaData *task, uint32_t *len)
{
    // 1. 入参合法性检查（输出任务指针不能为空）
    if (task == nullptr) {
        HCCL_VM_ERROR("[GetTaskCollection] 错误：输出任务指针 task 不能为空！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 TaskCollection 模块是否已初始化
    SHMManager::InitShm(false);
    TaskCollection *task_collection = SHMManager::GetTaskCollectionModule();
    if (task_collection == nullptr) {
        HCCL_VM_ERROR("[GetTaskCollection] 错误：TaskCollection 模块未初始化（请先调用 InitSharedMemory）！");
        return HcclVmResult::HCCL_SIM_SHM_NOT_INIT;
    }

    // 3. 核心读取逻辑（线程安全，依赖互斥锁保护）
    try {
        // 获取 TaskCollection 专属互斥锁（与插入函数共用同一把锁，保证读写安全）
        std::string mutex_name = std::string(SHM_MODULE_TASK_COLLECTION) + "_Mutex";
        ipc::interprocess_mutex *mutex = SHMManager::FindShmObject<ipc::interprocess_mutex>(mutex_name);
        if (!mutex) {
            throw std::runtime_error("未找到 TaskCollection 互斥锁");
        }

        // 加锁保护读取操作（避免读取时并发插入/修改）
        ipc::scoped_lock<ipc::interprocess_mutex> lock(*mutex);

        // 4. 获取已插入任务总数（indexGen 记录当前已插入任务数，从 0 到 indexGen-1 是有效任务）
        uint32_t task_count = task_collection->indexGen;
        if (task_count == 0) {
            HCCL_VM_INFO("[GetTaskCollection] 提示：TaskCollection 中无已插入任务！");
            return HcclVmResult::HCCL_SIM_SUCCESS;  // 无任务也算成功，避免误判
        }

        // 5. 遍历拷贝所有有效任务数据到输出指针
        // 注意：调用方需确保传入的 task 指针指向足够大的内存（至少 task_count 个 HcclTaskMetaData 大小）
        for (uint32_t i = 0; i < task_count; ++i) {
            task[i] = task_collection->tasks[i];  // 拷贝单个任务数据（POD 类型直接赋值）
        }

        *len = task_count;

        // 6. 日志输出（便于排查）
        HCCL_VM_INFO("[GetTaskCollection] 读取任务集合成功：");
        HCCL_VM_INFO("  - 已插入任务总数：{:d}", task_count);
        HCCL_VM_INFO("  - 输出任务指针：{:p}", static_cast<void*>(task));
        HCCL_VM_INFO("  - 第一个任务 rankId：{:d}", task[0].rankId);  // 示例输出第一个任务的 rankId
        HCCL_VM_INFO("  - 最后一个任务 rankId：{:d}", task[task_count - 1].rankId);

        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("[GetTaskCollection] 错误：读取任务集合失败 - {}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
}

HcclSim::HcclVmResult GetMemLayout(std::vector<DumpMemBlock> &memLayout)
{
    SHMManager::InitShm(false);

    MockProxyBuffer* mockBuffer = SHMManager::FindShmObject<MockProxyBuffer>(SHM_MODULE_MOCK_PROXY_BUFFER);

    if (mockBuffer == nullptr) {
        HCCL_VM_ERROR("GetMemLayout mockBuffer is nullptr");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
    uint64_t baseAddr = mockBuffer->mockBase;

    for (uint32_t i = 0; i < mockBuffer->mockMemCnt; i++) {
        MockBufferBlock bufferBlock = mockBuffer->mockMem[i];
        uint32_t rankId;
        uint8_t bufferType = mockBuffer->mockMem[i].bufferType;
        uint64_t addr = mockBuffer->mockMem[i].addr;
        if (addr > baseAddr) {
            addr -= baseAddr;
        } else {
            HCCL_VM_ERROR("GetMemLayout addr is invalid. baseAddr is {:x} and addr is {:x}", baseAddr, addr);
        }
        uint64_t size = mockBuffer->mockMem[i].size;

        HcclVmResult ret = GetRankIdByNpuPos(ShmNpuPos(mockBuffer->mockMem[i].npuPos), &rankId);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("GetMemLayout GetRankIdByNpuPos fail, npuPos is {}", ShmNpuPos(mockBuffer->mockMem[i].npuPos).ToString());
            return ret;
        }
        memLayout.push_back({rankId, addr, size, bufferType});
    }
    uint32_t npuNum = 0;
    HcclVmResult ret2 = GetNpuNum(&npuNum);
    if (ret2 != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("GetMemLayout GetNpuNum fail.");
        return ret2;
    }
    for (uint32_t i = 0; i < npuNum; i++) {
        ShmSimNpu* simNpu = nullptr;
        HcclVmResult ret = GetNpuByRankId(i, &simNpu);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("GetBlockSize GetNpuByRankId fail ");
            return ret;
        }
        for (int32_t j = 0; j < simNpu->memCount; j++) {
            uint64_t addr = reinterpret_cast<uint64_t>(simNpu->memory[j].addr.get());
            uint64_t size = simNpu->memory[j].size;
            uint8_t bufferType = simNpu->memory[j].bufferType;
            memLayout.push_back({i, addr, size, bufferType});
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclSim::HcclVmResult AllocatePhy(void **ptr, uint64_t *offset_ptr, size_t size)
{
    *ptr = sim::shm::ShmMemoryManager::GetInstance().AllocatePhy(size);
    *offset_ptr = sim::shm::ShmMemoryManager::GetInstance().GetHandleFromPtr(*ptr);
    if (*offset_ptr == 0) {
        HCCL_VM_ERROR("[AllocateOffset] Get Offset from ptr failed");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclSim::HcclVmResult DeallocatePhy(void *ptr, uint64_t offset_ptr, size_t size)
{
    void* tmpPtr = sim::shm::ShmMemoryManager::GetInstance().GetPtrFromHandle(offset_ptr);
    if (!sim::shm::ShmMemoryManager::GetInstance().DeallocatePhy(tmpPtr, size)) {
        HCCL_VM_ERROR("[DeallocateOffset] deallocate from offset failed offset:{:x}", offset_ptr);
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclSim::HcclVmResult GetPhyPtrFromOffsetPtr(void **ptr, uint64_t offset_ptr)
{
    *ptr = sim::shm::ShmMemoryManager::GetInstance().GetPtrFromHandle(offset_ptr);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclSim::HcclVmResult AllocateVir(void **ptr, size_t size)
{
    uint64_t offsetPtr = sim::shm::VirMemoryManager::GetInstance().AllocateVir(size);
    *ptr = (void *)offsetPtr;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclSim::HcclVmResult DeallocateVir(void *ptr, size_t size)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}