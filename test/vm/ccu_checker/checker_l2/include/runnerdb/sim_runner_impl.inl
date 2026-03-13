#ifndef SIM_RUNNER_IMPL_INL
#define SIM_RUNNER_IMPL_INL

#include "sim_runner_mem_db.h"
#include "sim_runner_db.h"

namespace RunnerDB {

    template <typename T>
    uint64_t Add(T& rec) {
        return SimRunnerMemDB::Instance().Add<T>(rec);
    }

    template <typename T>
    std::optional<T> GetById(uint64_t id) {
        return SimRunnerMemDB::Instance().Find<T>(id);
    }

    template <typename T>
    std::vector<T> GetByPred(std::function<bool(const T&)> pred) {
        return SimRunnerMemDB::Instance().QueryList<T>(pred);
    }

    template <typename T>
    std::pair<T, bool> GetOneByPred(std::function<bool(const T&)> pred) {
        return SimRunnerMemDB::Instance().Query<T>(pred);
    }

    template <typename T>
    bool Update(uint64_t id, std::function<void(T&)> updater) {
        return SimRunnerMemDB::Instance().Update<T>(id, updater);
    }

    template <typename T>
    bool Delete(uint64_t id) {
        return SimRunnerMemDB::Instance().Delete<T>(id);
    }

}

#endif