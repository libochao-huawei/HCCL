#include <iostream>
#include "sim_runner_ops.h"
#include "sim_runner_db.h"
#include "sim_models.h"
#include "sim_runner_common.h"
#include "hccl_vm_log.h"

namespace sim {

thread_local Runner g_runner;

thread_local uint64_t g_last_streamId;
thread_local uint64_t g_last_taskId;
thread_local int g_tsId = 0;
bool InsertRunner()
{
    auto host = RunnerDB::GetOneByPred<sim::Host>([](const sim::Host& h) {
        return true;
    });
    if (!host.second) {
        // not find
        HCCL_VM_ERROR("[InsertRunner] can not find any host");
        return false;
    }
    // 插入runner
    sim::Runner runner{};
    runner.host_id = host.first.id;
    runner.pid = getpid();
    runner.thread_id = pthread_self();
    
    // todo: runner未实际插入数据库，这里需要重新赋值
    g_runner.host_id = host.first.id;
    g_runner.pid = getpid();
    g_runner.thread_id = pthread_self();

    g_runner.id = RunnerDB::Add<sim::Runner>(runner);
    return true;
}


const Runner &GetCurrRunnerTls()
{
    if (g_runner.id == 0) {
        InsertRunner();
    }
    return g_runner;
}

bool SetCurrCtxTls(uint64_t ctx)
{
    auto &currRunnerId = g_runner.id;
    g_runner.current_ctx_id = ctx;
    RunnerDB::Update<sim::Runner>(currRunnerId,
                                   [currRunnerId, ctx](sim::Runner &runner) { runner.current_ctx_id = ctx; });

    return true;
}

void SetLastStreamIdTls(uint64_t streamId)
{
    g_last_streamId = streamId;
}

void SetLastTaskIdTls(uint64_t taskId)
{
    g_last_taskId = taskId;
}

uint64_t GetLastStreamIdTls()
{
    return g_last_streamId;
}

uint64_t GetLastTaskIdTls()
{
    return g_last_taskId;
}

uint64_t GetCurrRankId()
{
    auto currCtx = RunnerDB::GetById<sim::Context>(g_runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get CurrContext: {:d}", __func__, g_runner.current_ctx_id);
        return 0;
    }

    auto dev = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get device: {:d}", __func__, currCtx->device_id);
        return 0;
    }

    return dev->logic_id;
}

uint64_t GetRankIdByCtxId(uint64_t ctxId)
{
    auto currCtx = RunnerDB::GetById<sim::Context>(ctxId);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get CurrContext: {:d}", __func__, ctxId);
        return 0;
    }

    auto dev = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get device: {:d}", __func__, currCtx->device_id);
        return 0;
    }

    return dev->logic_id;
}

void SetTsDevice(int tsId)
{
    g_tsId = tsId;
}

uint32_t GetRankSize()
{
    auto allRank = RunnerDB::GetByPred<sim::Rank>([](const sim::Rank& r) {
        return true;
    });
    return allRank.size();
}

}
