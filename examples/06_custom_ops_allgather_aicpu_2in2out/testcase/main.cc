#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_allgather_2in2out.h"

#define ACLCHECK(expr)                                                                           \
    do {                                                                                         \
        auto _ret = (expr);                                                                      \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::cerr << "[ACL][ERROR] " << __FILE__ << ":" << __LINE__                       \
                      << " ret=" << _ret << std::endl;                                          \
            return static_cast<int>(_ret);                                                       \
        }                                                                                        \
    } while (0)

#define HCCLCHECK(expr)                                                                          \
    do {                                                                                         \
        auto _ret = (expr);                                                                      \
        if (_ret != HCCL_SUCCESS) {                                                              \
            std::cerr << "[HCCL][ERROR] " << __FILE__ << ":" << __LINE__                      \
                      << " ret=" << _ret << std::endl;                                          \
            return static_cast<int>(_ret);                                                       \
        }                                                                                        \
    } while (0)

namespace {

constexpr int kRouteValueStride = 100000;
constexpr int kRoute1BaseValue = 8000000;

struct Options {
    uint64_t count0 = 1024;
    uint64_t count1 = 2048;
    int warmup = 20;
    int iters = 100;
};

struct EnvContext {
    int rank = 0;
    int rankSize = 0;
    int deviceId = 0;
    bool mpiInited = false;
    bool aclInited = false;
    bool deviceSet = false;
    bool streamCreated = false;
    bool commCreated = false;
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
};

struct DeviceBuffers {
    void *send0 = nullptr;
    void *send1 = nullptr;
    void *recv0 = nullptr;
    void *recv1 = nullptr;
};

struct BenchmarkStats {
    double avgUs = 0.0;
    double bestUs = 0.0;
};

template<typename... Args>
void Log(int rank, Args &&...args)
{
    std::ostringstream oss;
    oss << "[Rank " << rank << "] ";
    (oss << ... << args);
    std::cout << oss.str() << std::endl;
}

bool ParseUint64(const std::string &text, uint64_t &value)
{
    char *end = nullptr;
    value = std::strtoull(text.c_str(), &end, 10);
    return end != nullptr && *end == '\0';
}

bool ParseInt(const std::string &text, int &value)
{
    char *end = nullptr;
    value = std::strtol(text.c_str(), &end, 10);
    return end != nullptr && *end == '\0';
}

bool ParseArgs(int argc, char *argv[], Options &options)
{
    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        auto readValue = [&](const std::string &prefix, auto &out) -> bool {
            if (arg.rfind(prefix, 0) != 0) {
                return false;
            }
            const std::string value = arg.substr(prefix.size());
            if constexpr (std::is_same_v<std::decay_t<decltype(out)>, uint64_t>) {
                return ParseUint64(value, out);
            } else {
                return ParseInt(value, out);
            }
        };

        bool ok = readValue("--count0=", options.count0) ||
                  readValue("--count1=", options.count1) ||
                  readValue("--warmup=", options.warmup) ||
                  readValue("--iters=", options.iters);
        if (!ok) {
            std::cerr << "Unknown or invalid arg: " << arg << std::endl;
            return false;
        }
    }
    return options.count0 > 0 && options.count1 > 0 && options.warmup >= 0 && options.iters > 0;
}

float BuildExpectedValue(int rank, uint64_t idx, bool isRoute1)
{
    // 这里特意把测试数据控制在 FP32 可精确表示的整数范围内，
    // 避免“通信结果是对的，但因为 float 精度丢失导致校验误报”。
    const int base = isRoute1 ? kRoute1BaseValue : 0;
    return static_cast<float>(base + rank * kRouteValueStride + static_cast<int>(idx));
}

int InitEnv(int argc, char *argv[], EnvContext &env)
{
    MPI_Init(&argc, &argv);
    env.mpiInited = true;
    MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &env.rankSize);

    ACLCHECK(aclInit(nullptr));
    env.aclInited = true;

    uint32_t deviceCount = 0;
    ACLCHECK(aclrtGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No Ascend device found" << std::endl;
        return -1;
    }

    env.deviceId = env.rank % static_cast<int>(deviceCount);
    ACLCHECK(aclrtSetDevice(env.deviceId));
    env.deviceSet = true;

    ACLCHECK(aclrtCreateStream(&env.stream));
    env.streamCreated = true;

    HcclRootInfo rootInfo;
    if (env.rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
    }
    MPI_Bcast(&rootInfo, sizeof(HcclRootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    HCCLCHECK(HcclCommInitRootInfo(env.rankSize, &rootInfo, env.rank, &env.comm));
    env.commCreated = true;
    return 0;
}

int AllocBuffers(const EnvContext &env, const Options &options, DeviceBuffers &bufs)
{
    const size_t sendBytes0 = options.count0 * sizeof(float);
    const size_t sendBytes1 = options.count1 * sizeof(float);
    const size_t recvBytes0 = options.count0 * static_cast<size_t>(env.rankSize) * sizeof(float);
    const size_t recvBytes1 = options.count1 * static_cast<size_t>(env.rankSize) * sizeof(float);

    ACLCHECK(aclrtMalloc(&bufs.send0, sendBytes0, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&bufs.send1, sendBytes1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&bufs.recv0, recvBytes0, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&bufs.recv1, recvBytes1, ACL_MEM_MALLOC_HUGE_FIRST));
    return 0;
}

int InitSendData(const EnvContext &env, const Options &options, DeviceBuffers &bufs)
{
    std::vector<float> hostSend0(options.count0);
    std::vector<float> hostSend1(options.count1);

    // route0 和 route1 故意使用两套不同模式，便于快速看出有没有串路。
    // 同时它们都落在 FP32 的精确整数区间里，校验时不会被精度误伤。
    for (uint64_t idx = 0; idx < options.count0; ++idx) {
        hostSend0[idx] = BuildExpectedValue(env.rank, idx, false);
    }
    for (uint64_t idx = 0; idx < options.count1; ++idx) {
        hostSend1[idx] = BuildExpectedValue(env.rank, idx, true);
    }

    ACLCHECK(aclrtMemcpy(bufs.send0, options.count0 * sizeof(float), hostSend0.data(),
        options.count0 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(bufs.send1, options.count1 * sizeof(float), hostSend1.data(),
        options.count1 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    return 0;
}

int ResetRecvBuffers(const EnvContext &env, const Options &options, DeviceBuffers &bufs)
{
    ACLCHECK(aclrtMemset(bufs.recv0, options.count0 * static_cast<size_t>(env.rankSize) * sizeof(float), 0,
        options.count0 * static_cast<size_t>(env.rankSize) * sizeof(float)));
    ACLCHECK(aclrtMemset(bufs.recv1, options.count1 * static_cast<size_t>(env.rankSize) * sizeof(float), 0,
        options.count1 * static_cast<size_t>(env.rankSize) * sizeof(float)));
    return 0;
}

int RunBaseline(const EnvContext &env, const Options &options, DeviceBuffers &bufs)
{
    HCCLCHECK(HcclAllGather(bufs.send0, bufs.recv0, options.count0, HCCL_DATA_TYPE_FP32, env.comm, env.stream));
    HCCLCHECK(HcclAllGather(bufs.send1, bufs.recv1, options.count1, HCCL_DATA_TYPE_FP32, env.comm, env.stream));
    ACLCHECK(aclrtSynchronizeStream(env.stream));
    return 0;
}

int RunFused(const EnvContext &env, const Options &options, DeviceBuffers &bufs)
{
    HCCLCHECK(HcclAllGather2In2OutAicpuCustom(
        bufs.send0, bufs.send1,
        bufs.recv0, bufs.recv1,
        options.count0, options.count1,
        HCCL_DATA_TYPE_FP32,
        env.comm, env.stream));
    ACLCHECK(aclrtSynchronizeStream(env.stream));
    return 0;
}

int CheckRouteResult(const EnvContext &env, uint64_t count, void *recvBuf, bool isRoute1)
{
    std::vector<float> hostRecv(count * static_cast<size_t>(env.rankSize));
    ACLCHECK(aclrtMemcpy(hostRecv.data(), hostRecv.size() * sizeof(float), recvBuf,
        hostRecv.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int rank = 0; rank < env.rankSize; ++rank) {
        for (uint64_t idx = 0; idx < count; ++idx) {
            const float expected = BuildExpectedValue(rank, idx, isRoute1);
            const float actual = hostRecv[static_cast<size_t>(rank) * count + idx];
            if (std::fabs(actual - expected) > 1e-5f) {
                std::cerr << "Verify failed: route=" << (isRoute1 ? 1 : 0)
                          << " rankSlice=" << rank
                          << " idx=" << idx
                          << " expected=" << expected
                          << " actual=" << actual << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

int VerifyResults(const EnvContext &env, const Options &options, DeviceBuffers &bufs, const char *tag)
{
    if (CheckRouteResult(env, options.count0, bufs.recv0, false) != 0) {
        Log(env.rank, tag, " route0 校验失败");
        return -1;
    }
    if (CheckRouteResult(env, options.count1, bufs.recv1, true) != 0) {
        Log(env.rank, tag, " route1 校验失败");
        return -1;
    }
    Log(env.rank, tag, " route0/route1 校验通过");
    return 0;
}

template<typename RunFunc>
int BenchmarkPath(const EnvContext &env, const Options &options, DeviceBuffers &bufs,
    const char *pathName, RunFunc runFunc, BenchmarkStats &stats)
{
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.iters));

    const int totalRounds = options.warmup + options.iters;
    for (int round = 0; round < totalRounds; ++round) {
        if (ResetRecvBuffers(env, options, bufs) != 0) {
            return -1;
        }
        ACLCHECK(aclrtSynchronizeStream(env.stream));
        MPI_Barrier(MPI_COMM_WORLD);

        const auto begin = std::chrono::high_resolution_clock::now();
        if (runFunc() != 0) {
            return -1;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const auto end = std::chrono::high_resolution_clock::now();

        const double localUs = std::chrono::duration<double, std::micro>(end - begin).count();
        double maxUs = 0.0;
        MPI_Allreduce(&localUs, &maxUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (round >= options.warmup) {
            samples.push_back(maxUs);
        }
    }

    stats.avgUs = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    stats.bestUs = *std::min_element(samples.begin(), samples.end());
    Log(env.rank, pathName, " avgUs=", std::fixed, std::setprecision(2), stats.avgUs,
        " bestUs=", stats.bestUs);
    return 0;
}

void ReleaseBuffers(DeviceBuffers &bufs)
{
    if (bufs.send0 != nullptr) {
        aclrtFree(bufs.send0);
        bufs.send0 = nullptr;
    }
    if (bufs.send1 != nullptr) {
        aclrtFree(bufs.send1);
        bufs.send1 = nullptr;
    }
    if (bufs.recv0 != nullptr) {
        aclrtFree(bufs.recv0);
        bufs.recv0 = nullptr;
    }
    if (bufs.recv1 != nullptr) {
        aclrtFree(bufs.recv1);
        bufs.recv1 = nullptr;
    }
}

void FinalizeEnv(EnvContext &env)
{
    if (env.commCreated && env.comm != nullptr) {
        HcclCommDestroy(env.comm);
        env.comm = nullptr;
        env.commCreated = false;
    }
    if (env.streamCreated && env.stream != nullptr) {
        aclrtDestroyStream(env.stream);
        env.stream = nullptr;
        env.streamCreated = false;
    }
    if (env.deviceSet) {
        aclrtResetDevice(env.deviceId);
        env.deviceSet = false;
    }
    if (env.aclInited) {
        aclFinalize();
        env.aclInited = false;
    }
    if (env.mpiInited) {
        MPI_Finalize();
        env.mpiInited = false;
    }
}

double SafeSpeedup(double baseline, double fused)
{
    return (fused <= 0.0) ? 0.0 : (baseline / fused);
}

} // namespace

int main(int argc, char *argv[])
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        std::cerr << "Usage: ./main [--count0=1024] [--count1=2048] [--warmup=20] [--iters=100]" << std::endl;
        return -1;
    }

    EnvContext env;
    DeviceBuffers baselineBufs;
    DeviceBuffers fusedBufs;

    int ret = InitEnv(argc, argv, env);
    if (ret != 0) {
        FinalizeEnv(env);
        return ret;
    }

    Log(env.rank, "配置: count0=", options.count0,
        " count1=", options.count1,
        " warmup=", options.warmup,
        " iters=", options.iters);

    do {
        if (AllocBuffers(env, options, baselineBufs) != 0 || AllocBuffers(env, options, fusedBufs) != 0) {
            ret = -1;
            break;
        }
        if (InitSendData(env, options, baselineBufs) != 0 || InitSendData(env, options, fusedBufs) != 0) {
            ret = -1;
            break;
        }

        // 先做功能校验，再做 benchmark，这样后面看到性能数据时心里有底。
        if (ResetRecvBuffers(env, options, baselineBufs) != 0 || RunBaseline(env, options, baselineBufs) != 0) {
            ret = -1;
            break;
        }
        if (VerifyResults(env, options, baselineBufs, "baseline") != 0) {
            ret = -1;
            break;
        }

        if (ResetRecvBuffers(env, options, fusedBufs) != 0 || RunFused(env, options, fusedBufs) != 0) {
            ret = -1;
            break;
        }
        if (VerifyResults(env, options, fusedBufs, "fused") != 0) {
            ret = -1;
            break;
        }

        BenchmarkStats baselineStats;
        BenchmarkStats fusedStats;
        if (BenchmarkPath(env, options, baselineBufs, "baseline",
                [&]() { return RunBaseline(env, options, baselineBufs); }, baselineStats) != 0) {
            ret = -1;
            break;
        }
        if (BenchmarkPath(env, options, fusedBufs, "fused",
                [&]() { return RunFused(env, options, fusedBufs); }, fusedStats) != 0) {
            ret = -1;
            break;
        }

        if (env.rank == 0) {
            std::cout << "[Summary] baseline_avg_us=" << baselineStats.avgUs
                      << " fused_avg_us=" << fusedStats.avgUs
                      << " speedup=" << SafeSpeedup(baselineStats.avgUs, fusedStats.avgUs) << std::endl;
            std::cout << "[Summary] baseline_best_us=" << baselineStats.bestUs
                      << " fused_best_us=" << fusedStats.bestUs
                      << " speedup_best=" << SafeSpeedup(baselineStats.bestUs, fusedStats.bestUs) << std::endl;
        }
    } while (false);

    ReleaseBuffers(baselineBufs);
    ReleaseBuffers(fusedBufs);
    FinalizeEnv(env);
    return ret;
}
