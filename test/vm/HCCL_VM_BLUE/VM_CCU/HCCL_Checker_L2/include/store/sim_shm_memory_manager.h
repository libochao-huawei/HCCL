#ifndef SIM_SHM_MEMORY_MANAGER_H
#define SIM_SHM_MEMORY_MANAGER_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <memory>
#include <string>
#include <functional>
#include <mutex>
#include <stdio.h>
#include <stdexcept>
#include <cstdint>
#include <atomic>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstddef>
#include <utility>

namespace sim {
namespace shm {

namespace bip = boost::interprocess;

struct ShmParams {
    std::string name;
    uint64_t size;

    ShmParams(const std::string &n = "", uint64_t s = 0) : name(n), size(s)
    {
        if (name.empty()) {
            throw std::invalid_argument("[ShmParams] name cannot be empty");
        }
        if (size == 0) {
            throw std::invalid_argument("[ShmParams] size cannot be 0");
        }
    }
};

struct InitParams {
    ShmParams db;
    ShmParams pool;
    bool create;
    InitParams()
        : db("SimShmDB", static_cast<uint64_t>(512) * 1024 * 1024),
          pool("SimShmPool", static_cast<uint64_t>(4) * 1024 * 1024 * 1024),
          create(false)
    {
    }
};

class ShmMemoryManager {
public:
    using AllocatorType = bip::allocator<char, bip::managed_shared_memory::segment_manager>;
    ShmMemoryManager();
    ~ShmMemoryManager();

    static bool Initialize(bool create)
    {
        std::lock_guard<std::mutex> lock(s_initMtx);
        if (s_isInitialized) {
            std::cerr << "[ShmMemoryManager] Already initialized" << std::endl;
            return false;
        }

        auto ret = GetInstanceInternal().Init(create);
        if (!ret) {
            std::cerr << "[ShmMemoryManager] Failed to initialize shared memory segments" << std::endl;
            return false;
        }
        s_isInitialized = true;
        return true;
    }

    static ShmMemoryManager &GetInstance()
    {
        if (!s_isInitialized) {
            Initialize(false);
        }
        return GetInstanceInternal();
    }

    void *AllocatePhy(size_t size)
    {
        if (size == 0) {
            std::cerr << "[ShmMemoryManager] AllocatePhy size cannot be 0" << std::endl;
            return 0;
        }

        std::lock_guard<std::mutex> lock(m_poolMtx);
        if (!m_poolSegment || !m_poolMutex || !m_poolAlloc) {
            std::cerr << "[ShmMemoryManager] Pool segment not initialized" << std::endl;
            return 0;
        }

        try {
            bip::scoped_lock<bip::interprocess_mutex> ipc_lock(*m_poolMutex);
            bip::offset_ptr<char> offsetPtr = m_poolAlloc->allocate(size);
            if (!offsetPtr) {
                std::cerr << "[ShmMemoryManager] AllocatePhy failed (out of memory, size=" << size << ")" << std::endl;
                return 0;
            }

            return offsetPtr.get();
        } catch (const bip::bad_alloc &e) {
            std::cerr << "[ShmMemoryManager] Bad alloc (size=" << size << "): " << e.what() << std::endl;
            return 0;
        } catch (const bip::interprocess_exception &e) {
            std::cerr << "[ShmMemoryManager] Boost exception allocate: " << e.what() << std::endl;
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "[ShmMemoryManager] Exception allocate: " << e.what() << std::endl;
            return 0;
        }
    }

    bool DeallocatePhy(void *ptr, size_t size)
    {
        if (ptr == nullptr || size == 0) {
            std::cerr << "[ShmMemoryManager] Invalid deallocate params (handle=" << ptr << ", size=" << size << ")"
                      << std::endl;
            return false;
        }

        std::lock_guard<std::mutex> lock(m_poolMtx);
        if (!m_poolSegment || !m_poolMutex || !m_poolAlloc) {
            std::cerr << "[ShmMemoryManager] Pool segment not initialized" << std::endl;
            return false;
        }

        try {
            bip::scoped_lock<bip::interprocess_mutex> ipc_lock(*m_poolMutex);
            bip::offset_ptr<char> offset_ptr(static_cast<char *>(ptr));
            m_poolAlloc->deallocate(offset_ptr, size);
            return true;
        } catch (const bip::interprocess_exception &e) {
            std::cerr << "[ShmMemoryManager] Boost exception deallocate " << e.what() << std::endl;
            return false;
        } catch (const std::exception &e) {
            std::cerr << "[ShmMemoryManager] Exception deallocate (handle=" << ptr << "): " << e.what() << std::endl;
            return false;
        }
    }

    uint64_t GetHandleFromPtr(void *offsetPtr)
    {
        if (!offsetPtr || m_poolSegment == nullptr) {
            throw std::invalid_argument("[ShmMemoryManager] Invalid ptr or uninitialized pool segment");
        }
        uintptr_t handle = m_poolSegment->get_handle_from_address(offsetPtr);
        return static_cast<uint64_t>(handle);
    }

    void *GetPtrFromHandle(uint64_t handle)
    {
        if (m_poolSegment == nullptr) {
            throw std::runtime_error("[ShmMemoryManager] Pool segment not initialized");
        }
        void *restored_ptr =
            static_cast<void *>(m_poolSegment->get_address_from_handle(static_cast<uintptr_t>(handle)));
        if (!restored_ptr) {
            throw std::invalid_argument("[ShmMemoryManager] Invalid handle: " + std::to_string(handle));
        }
        return restored_ptr;
    }

    bip::managed_shared_memory *GetDbSegment()
    {
        std::lock_guard<std::mutex> lock(m_dbMtx);
        if (m_dbSegment == nullptr) {
            throw std::runtime_error("[ShmMemoryManager] DB segment not initialized");
        }
        return m_dbSegment;
    }

    bip::managed_shared_memory *GetPoolSegment()
    {
        std::lock_guard<std::mutex> lock(m_dbMtx);
        if (m_poolSegment == nullptr) {
            throw std::runtime_error("[ShmMemoryManager] Pool segment not initialized");
        }
        return m_poolSegment;
    }

    static ShmMemoryManager &GetInstanceInternal();
    bool Init(bool create);

    bool CreateDBShm();
    bool OpenDBShm();
    bool CreatePoolShm();
    bool OpenPoolShm();

private:
    static std::atomic<bool> s_isInitialized;
    static std::mutex s_initMtx;

    bool m_isCreated{false};
    std::mutex m_dbMtx;
    std::mutex m_poolMtx;

    bip::managed_shared_memory *m_dbSegment{nullptr};
    bip::managed_shared_memory *m_poolSegment{nullptr};
    bip::interprocess_mutex *m_poolMutex{nullptr};
    AllocatorType *m_poolAlloc{nullptr};
};

}  // namespace shm
}  // namespace sim

#endif  // SIM_SHM_MEMORY_MANAGER_H