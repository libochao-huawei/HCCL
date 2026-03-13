#ifndef THREAD_SAFE_MAP_H
#define THREAD_SAFE_MAP_H

#include <map>
#include <mutex>

template<typename K, typename V>
class ThreadSafeMap {
public:
    V& operator[](const K& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_[key];
    }

    void insert(const K& key, const V& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_[key] = value;
    }

    bool try_get(const K& key, V& output) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = data_.find(key);
        if (it != data_.end()) {
            output = it->second;
            return true;
        }
        return false;
    }

    bool erase(const K& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.erase(key) > 0;
    }

    bool contains(const K& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.find(key) != data_.end();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
    }

    // 遍历操作
    template<typename F>
    void for_each(F func) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : data_) {
            func(pair.first, pair.second);
        }
    }

    // 条件删除
    template<typename Predicate>
    size_t erase_if(Predicate pred) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (auto it = data_.begin(); it != data_.end(); ) {
            if (pred(it->first, it->second)) {
                it = data_.erase(it);
                ++count;
            } else {
                ++it;
            }
        }
        return count;
    }

private:
    std::map<K, V> data_;
    mutable std::mutex mutex_;
};

#endif //THREAD_SAFE_MAP_H
