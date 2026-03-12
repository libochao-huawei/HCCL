#ifndef SIM_RUNNER_DB_H
#define SIM_RUNNER_DB_H

#include <optional>
#include <vector>
#include <functional>
#include "sim_models.h"

namespace RunnerDB {

    template <typename T>
    uint64_t Add(T& rec);

    template <typename T>
    std::optional<T> GetById(uint64_t id);

    template <typename T>
    std::vector<T> GetByPred(std::function<bool(const T&)> pred);

    template <typename T>
    std::pair<T, bool> GetOneByPred(std::function<bool(const T&)> pred);

    template <typename T>
    bool Update(uint64_t id, std::function<void(T&)> updater);

    template <typename T>
    bool Delete(uint64_t id);

}

#include "sim_runner_impl.inl"

#endif