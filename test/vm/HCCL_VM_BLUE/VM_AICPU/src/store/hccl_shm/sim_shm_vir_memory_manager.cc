#include <store/sim_shm_vir_memory_manager.h>

namespace sim {
namespace shm {

VirMemoryManager& VirMemoryManager::GetInstance()
{
    static std::mutex s_mtx;
    static VirMemoryManager* s_instance = nullptr;
    std::lock_guard<std::mutex> lock(s_mtx);
    if (s_instance == nullptr) {
        s_instance = new VirMemoryManager();
    }
    return *s_instance;
}

VirMemoryManager::VirMemoryManager()
{

}

VirMemoryManager::~VirMemoryManager()
{

}

bool VirMemoryManager::Initialize(bool create)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    auto segment = sim::shm::ShmMemoryManager::GetInstance().GetPoolSegment();
    std::string counterName = m_virMemName;
    std::string mutexName = m_virMemName + "_mtx";
    if (create) {
        m_virMemMutex = segment->find_or_construct<bip::interprocess_mutex>(mutexName.c_str())();
        if (!m_virMemMutex) {
            return false;
        }
        
        m_virMemStart = segment->find_or_construct<uint64_t>(counterName.c_str())(1);
        if (!m_virMemStart) {
            segment->destroy<bip::interprocess_mutex>(mutexName.c_str());
            return false;
        }
    } else {
        auto mutexResult = segment->find<bip::interprocess_mutex>(mutexName.c_str());
        if (!mutexResult.first) {
            return false;
        }
        m_virMemMutex = mutexResult.first;
        
        auto counterResult = segment->find<uint64_t>(counterName.c_str());
        if (!counterResult.first) {
            return false;
        }
        m_virMemStart = counterResult.first;
    }
    isInit = true;
    return true;
}

uint64_t VirMemoryManager::AllocateVir(uint64_t size)
{   
    if (!isInit) {
        Initialize(false);
    }

    bip::scoped_lock<bip::interprocess_mutex> lock(*m_virMemMutex);
    uint64_t ret = *m_virMemStart;
    *m_virMemStart += size;
    return ret;
}

void VirMemoryManager::DeallocateVir(uint64_t offsetPtr)
{
    return;
}

}
}