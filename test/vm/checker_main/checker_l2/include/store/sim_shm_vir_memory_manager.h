
#ifndef VIR_MEMOROY_MENAGER_H
#define VIR_MEMOROY_MENAGER_H

#include <mutex>
#include <store/sim_shm_memory_manager.h>

namespace sim {
namespace shm {
class VirMemoryManager {
public:
    static VirMemoryManager &GetInstance();
    VirMemoryManager(const VirMemoryManager &) = delete;
    VirMemoryManager &operator=(const VirMemoryManager &) = delete;

    bool Initialize(bool create);

    uint64_t AllocateVir(uint64_t size);
    void DeallocateVir(uint64_t offsetPtr);
private:
    VirMemoryManager();
    ~VirMemoryManager();

    std::mutex m_mutex;
    std::string m_virMemName{"simVirMemroy"};
    bip::interprocess_mutex *m_virMemMutex{nullptr};
    uint64_t *m_virMemStart{nullptr};
    bool isInit{false};
};
}
}
#endif