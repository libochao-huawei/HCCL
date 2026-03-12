#ifndef SIM_MEM_TABLE_H
#define SIM_MEM_TABLE_H
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <map>
#include <functional>
#include <optional>
#include <atomic>
#include <unordered_map>
#include <stdio.h>

#include <store/sim_shm_memory_manager.h>

namespace bip = boost::interprocess;

template <typename, typename = void>
struct has_id_field : std::false_type {
};

template <typename T>
struct has_id_field<T, decltype((void)std::declval<T &>().id, void())> : std::true_type {
};

template <typename T>
typename std::enable_if<has_id_field<T>::value>::type setId(T &obj, uint64_t id)
{
    obj.id = id;
}

template <typename T>
typename std::enable_if<!has_id_field<T>::value>::type setId(T &, uint64_t)
{
}

template <typename T, bool Shared = true>
class Table {
public:
    using KeyType = uint64_t;
    using ValueType = T;
    using ValueTypePair = std::pair<const KeyType, ValueType>;
    using ShmAllocator = bip::allocator<ValueTypePair, bip::managed_shared_memory::segment_manager>;
    using ShmMap = bip::map<KeyType, ValueType, std::less<KeyType>, ShmAllocator>;

    Table(const Table &) = delete;
    Table &operator=(const Table &) = delete;
    Table &operator=(Table &&other) = delete;
    Table(Table &&other) = delete;
    ~Table() = default;

    Table(const std::string &tableName = "") : m_tableName(tableName), m_mapShm(nullptr), m_localIdCounter(1)
    {
        if constexpr (Shared) {
            auto segment = sim::shm::ShmMemoryManager::GetInstance().GetDbSegment();
            ShmAllocator allocator(segment->get_segment_manager());
            m_mapShm = segment->find_or_construct<ShmMap>(m_tableName.c_str())(std::less<KeyType>(), allocator);

            std::string mutexName = m_tableName + "_mutex";
            m_mapShmMutex = segment->find_or_construct<bip::interprocess_mutex>(mutexName.c_str())();

            std::string counterName = m_tableName + "_id_counter";
            m_idCounterPtr = segment->find_or_construct<KeyType>(counterName.c_str())(1);
        }
    }

    uint64_t Add(T &rec)
    {
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            KeyType newId = *m_idCounterPtr;
            (*m_idCounterPtr)++;
            setId(rec, newId);
            auto [it, success] = m_mapShm->insert(std::make_pair(newId, rec));
            if (!success) {
                return 0;
            }
            return newId;
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            KeyType newId = m_localIdCounter.fetch_add(1, std::memory_order_relaxed);
            setId(rec, newId);
            auto [it, success] = m_mapLocal.insert(std::make_pair(newId, rec));
            if (!success) {
                return 0;
            }
            return newId;
        }
    }

    bool Update(uint64_t id, std::function<void(T &)> updater)
    {
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            auto it = m_mapShm->find(id);
            if (it == m_mapShm->end()) {
                return false;
            }
            updater(it->second);
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            auto it = m_mapLocal.find(id);
            if (it == m_mapLocal.end()) {
                return false;
            }
            updater(it->second);
        }
        return true;
    }

    bool Delete(uint64_t id)
    {
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            return m_mapShm->erase(id) > 0;
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            return m_mapLocal.erase(id) > 0;
        }
    }

    bool DeleteAll()
    {
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            m_mapShm->clear();
            return true;
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            m_mapLocal.clear();
            return true;
        }
    }

    std::optional<ValueType> Find(uint64_t id) const
    {
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            auto it = m_mapShm->find(id);
            if (it == m_mapShm->end()) {
                return std::nullopt;
            }
            return it->second;
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            auto it = m_mapLocal.find(id);
            if (it == m_mapLocal.end()) {
                return std::nullopt;
            }
            return it->second;
        }
    }

    const ValueType* FindPtr(uint64_t id) const
    {
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            auto it = m_mapShm->find(id);
            return (it == m_mapShm->end()) ? nullptr : &(it->second);
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            auto it = m_mapLocal.find(id);
            return (it == m_mapLocal.end()) ? nullptr : &(it->second);
        }
    }

    std::vector<ValueType> QueryList(std::function<bool(const T &)> pred) const
    {
        std::vector<ValueType> result;
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            for (const auto &pair : *m_mapShm) {
                if (pred(pair.second)) {
                    result.push_back(pair.second);
                }
            }
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            for (const auto &pair : m_mapLocal) {
                if (pred(pair.second)) {
                    result.push_back(pair.second);
                }
            }
        }
        return result;
    }

    std::pair<ValueType, bool> Query(std::function<bool(const T &)> pred) const
    {
        if constexpr (Shared) {
            bip::scoped_lock<bip::interprocess_mutex> lock(*m_mapShmMutex);
            for (const auto &pair : *m_mapShm) {
                if (pred(pair.second)) {
                    return {pair.second, true};
                }
            }
        } else {
            std::lock_guard<std::mutex> lock(m_mapLocalMutex);
            for (const auto &pair : m_mapLocal) {
                if (pred(pair.second)) {
                    return {pair.second, true};
                }
            }
        }
        return {ValueType{}, false};
    }

    std::string GetTableName() const {
        return m_tableName;
    }
private:
    std::string m_tableName;
    ShmMap *m_mapShm;
    bip::interprocess_mutex *m_mapShmMutex = nullptr;
    KeyType *m_idCounterPtr = nullptr;
    // local
    std::map<KeyType, ValueType> m_mapLocal;
    std::mutex m_mapLocalMutex;
    std::atomic<KeyType> m_localIdCounter;
};

#endif