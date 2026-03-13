#ifndef SIM_RUNNER_IMPL_INL
#define SIM_RUNNER_IMPL_INL

#include "sim_sqlite_db.h"
#include <optional>
#include "sim_runner_db.h"

namespace RunnerDB {

    template <typename T>
    uint64_t Add(T& rec) {
        return SimRunnerSqliteDB::Instance().Add<T>(rec);
    }

    template <typename T>
    std::optional<T> GetById(uint64_t id) {
        return SimRunnerSqliteDB::Instance().Find<T>(id);
    }

    template <typename T>
    std::vector<T> GetByPred(std::function<bool(const T&)> pred) {
        return SimRunnerSqliteDB::Instance().QueryList<T>(pred);
    }

    template <typename T>
    std::pair<T, bool> GetOneByPred(std::function<bool(const T&)> pred) {
        return SimRunnerSqliteDB::Instance().Query<T>(pred);
    }

    template <typename T>
    bool Update(uint64_t id, std::function<void(T&)> updater) {
        return SimRunnerSqliteDB::Instance().Update<T>(id, updater);
    }

    template <typename T>
    bool Delete(uint64_t id) {
        return SimRunnerSqliteDB::Instance().Delete<T>(id);
    }

    template <typename T>
    bool DeleteAll() {
        return SimRunnerSqliteDB::Instance().DeleteAll<T>();
    }
}

#endif