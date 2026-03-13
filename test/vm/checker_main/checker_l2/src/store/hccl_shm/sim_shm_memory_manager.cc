#include <mutex>
#include <store/sim_shm_memory_manager.h>
#include "hccl_vm_log.h"

namespace sim {
namespace shm {
std::atomic<bool> ShmMemoryManager::s_isInitialized{false};
std::mutex ShmMemoryManager::s_initMtx;

InitParams g_shmMemParams;

ShmMemoryManager::ShmMemoryManager()
{
}
ShmMemoryManager::~ShmMemoryManager()
{
    if (m_dbSegment) {
        delete m_dbSegment;
        m_dbSegment = nullptr;
    }

    if (m_poolAlloc) {
        delete m_poolAlloc;
        m_poolAlloc = nullptr;
    }
    m_poolMutex = nullptr;
    if (m_poolSegment) {
        delete m_poolSegment;
        m_poolSegment = nullptr;
    }

    if (m_isCreated) {
        bip::shared_memory_object::remove(g_shmMemParams.db.name.c_str());
        bip::shared_memory_object::remove(g_shmMemParams.pool.name.c_str());
    }
}

bool ShmMemoryManager::CreateDBShm()
{
    bip::shared_memory_object::remove(g_shmMemParams.db.name.c_str());
    m_dbSegment = new bip::managed_shared_memory(bip::create_only, g_shmMemParams.db.name.c_str(), g_shmMemParams.db.size);

    return true;
}
bool ShmMemoryManager::OpenDBShm()
{
    m_dbSegment = new bip::managed_shared_memory(bip::open_only, g_shmMemParams.db.name.c_str());
    if (m_dbSegment == nullptr) {
        HCCL_VM_ERROR("[{}] Opened db shared memory: {}", __func__, g_shmMemParams.db.name);
        return false;
    }
    return true;
}

bool ShmMemoryManager::CreatePoolShm()
{
    bip::shared_memory_object::remove(g_shmMemParams.pool.name.c_str());
    m_poolSegment = new bip::managed_shared_memory(bip::create_only, g_shmMemParams.pool.name.c_str(), g_shmMemParams.pool.size);

    std::string mutexName = g_shmMemParams.pool.name + "_MTX";
    m_poolMutex = m_poolSegment->find_or_construct<bip::interprocess_mutex>(mutexName.c_str())();
    if (m_poolMutex == nullptr) {
        HCCL_VM_ERROR("[{}] Failed to construct mutex {}", __func__, mutexName);
        return false;
    }
    return true;
}

bool ShmMemoryManager::OpenPoolShm()
{
    m_poolSegment = new bip::managed_shared_memory(bip::open_only, g_shmMemParams.pool.name.c_str());
    if (m_poolSegment == nullptr) {
        HCCL_VM_ERROR("[{}] Opened shared memory failed {}", __func__, g_shmMemParams.pool.name);
        return false;
    }
    std::string mutexName = g_shmMemParams.pool.name + "_MTX";
    auto mutexRet = m_poolSegment->find<bip::interprocess_mutex>(mutexName.c_str());
    m_poolMutex = mutexRet.first;
    if (m_poolMutex == nullptr) {
        HCCL_VM_ERROR("[{}] Failed to open segment {}", __func__, mutexName);
        return false;
    }
    return true;
}

ShmMemoryManager& ShmMemoryManager::GetInstanceInternal()
{
    static std::mutex s_mtx;
    static ShmMemoryManager* s_instance = nullptr;
    std::lock_guard<std::mutex> lock(s_mtx);
    if (s_instance == nullptr) {
        s_instance = new ShmMemoryManager();
    }
    return *s_instance;
}

bool ShmMemoryManager::Init(bool create)
{
    if (create) {
        if (!CreateDBShm()) {
            throw std::runtime_error("Initialize DB segment failed");
        }

        if (!CreatePoolShm()) {
            throw std::runtime_error("Initialize Pool segment failed");
        }
    } else {
        if (!OpenDBShm()) {
            throw std::runtime_error("Open DB segment failed");
        }

        if (!OpenPoolShm()) {
            throw std::runtime_error("Open Pool segment failed");
        }
    }

    m_poolAlloc = new AllocatorType(m_poolSegment->get_segment_manager());
    if (!m_poolAlloc) {
        throw std::runtime_error("Initialize Pool allocator failed");
    }
    m_isCreated = create;
    HCCL_VM_INFO("[ShmMemoryManager] All segments and resources initialized");
    return true;
}

}
}