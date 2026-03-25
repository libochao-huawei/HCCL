#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_double_allgather.h"
#include "common.h"

#define ACLCHECK(expr)                                                                          \
    do {                                                                                        \
        auto _ret = (expr);                                                                     \
        if (_ret != ACL_SUCCESS) {                                                              \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                        \
        }                                                                                       \
    } while (0)

#define HCCLCHECK(expr)                                                                         \
    do {                                                                                        \
        auto _ret = (expr);                                                                     \
        if (_ret != HCCL_SUCCESS) {                                                             \
            printf("[ERROR] hccl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                        \
        }                                                                                       \
    } while (0)

using namespace ops_hccl_double_allgather;

struct Options {
    uint64_t count0 = 1024;
    uint64_t count1 = 2048;
    HcclDataType dtype0 = HCCL_DATA_TYPE_FP32;
    HcclDataType dtype1 = HCCL_DATA_TYPE_INT32;
    int warmup = 5;
    int iters = 20;
};

struct RuntimeContext {
    int rank = 0;
    int worldSize = 0;
    int deviceId = 0;
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    void *sendBuf0 = nullptr;
    void *recvBuf0 = nullptr;
    void *sendBuf1 = nullptr;
    void *recvBuf1 = nullptr;
};

static std::string DataTypeToString(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_FP16:
            return "fp16";
        case HCCL_DATA_TYPE_FP32:
            return "fp32";
        case HCCL_DATA_TYPE_INT32:
            return "int32";
        default:
            return "unknown";
    }
}

static bool ParseDataType(const std::string &text, HcclDataType &dataType)
{
    if (text == "fp16") {
        dataType = HCCL_DATA_TYPE_FP16;
        return true;
    }
    if (text == "fp32") {
        dataType = HCCL_DATA_TYPE_FP32;
        return true;
    }
    if (text == "int32") {
        dataType = HCCL_DATA_TYPE_INT32;
        return true;
    }
    return false;
}

static void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [--count0 N] [--count1 N] [--dtype0 fp16|fp32|int32]"
              << " [--dtype1 fp16|fp32|int32] [--warmup N] [--iters N]" << std::endl;
}

static bool ParseArgs(int argc, char *argv[], Options &options)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char *name) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                return false;
            }
            return true;
        };
        if (arg == "--count0") {
            if (!needValue("--count0")) return false;
            options.count0 = std::stoull(argv[++i]);
        } else if (arg == "--count1") {
            if (!needValue("--count1")) return false;
            options.count1 = std::stoull(argv[++i]);
        } else if (arg == "--dtype0") {
            if (!needValue("--dtype0")) return false;
            if (!ParseDataType(argv[++i], options.dtype0)) return false;
        } else if (arg == "--dtype1") {
            if (!needValue("--dtype1")) return false;
            if (!ParseDataType(argv[++i], options.dtype1)) return false;
        } else if (arg == "--warmup") {
            if (!needValue("--warmup")) return false;
            options.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters") {
            if (!needValue("--iters")) return false;
            options.iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return options.count0 > 0 && options.count1 > 0 && options.warmup >= 0 && options.iters > 0;
}

static void FillPattern(std::vector<uint8_t> &buffer, uint64_t count, HcclDataType dataType, int rank, int gatherId)
{
    if (dataType == HCCL_DATA_TYPE_FP16) {
        auto *typed = reinterpret_cast<uint16_t *>(buffer.data());
        for (uint64_t i = 0; i < count; ++i) {
            typed[i] = static_cast<uint16_t>((gatherId * 100 + rank * 10 + static_cast<int>(i % 10)) & 0x7FFF);
        }
        return;
    }

    auto *typed = reinterpret_cast<uint32_t *>(buffer.data());
    for (uint64_t i = 0; i < count; ++i) {
        typed[i] = static_cast<uint32_t>(gatherId * 100000 + rank * 1000 + static_cast<int>(i));
    }
}

static void BuildExpected(std::vector<uint8_t> &buffer, uint64_t count, HcclDataType dataType, int worldSize, int gatherId)
{
    size_t elemSize = GetElemBytes(dataType);
    size_t singleBytes = static_cast<size_t>(count) * elemSize;
    for (int rank = 0; rank < worldSize; ++rank) {
        std::vector<uint8_t> tmp(singleBytes);
        FillPattern(tmp, count, dataType, rank, gatherId);
        std::memcpy(buffer.data() + rank * singleBytes, tmp.data(), singleBytes);
    }
}

static int InitEnv(int argc, char *argv[], RuntimeContext &ctx)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.worldSize);

    ACLCHECK(aclInit(nullptr));
    uint32_t devCount = 0;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    if (devCount == 0 || ctx.worldSize > static_cast<int>(devCount)) {
        if (ctx.rank == 0) {
            std::cerr << "Available devices are not enough for current MPI ranks" << std::endl;
        }
        return -1;
    }

    ctx.deviceId = ctx.rank % static_cast<int>(devCount);
    ACLCHECK(aclrtSetDevice(ctx.deviceId));
    ACLCHECK(aclrtCreateStream(&ctx.stream));

    HcclRootInfo rootInfo;
    if (ctx.rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
    }
    MPI_Bcast(&rootInfo, sizeof(HcclRootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    HCCLCHECK(HcclCommInitRootInfo(ctx.worldSize, &rootInfo, ctx.rank, &ctx.comm));
    return 0;
}

static int PrepareBuffers(RuntimeContext &ctx, const Options &options)
{
    size_t sendBytes0 = 0;
    size_t recvBytes0 = 0;
    size_t sendBytes1 = 0;
    size_t recvBytes1 = 0;
    if (!GetBytesByCount(options.count0, options.dtype0, sendBytes0) ||
        !GetBytesByCount(options.count1, options.dtype1, sendBytes1)) {
        return -1;
    }
    recvBytes0 = sendBytes0 * static_cast<size_t>(ctx.worldSize);
    recvBytes1 = sendBytes1 * static_cast<size_t>(ctx.worldSize);

    ACLCHECK(aclrtMalloc(&ctx.sendBuf0, sendBytes0, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&ctx.recvBuf0, recvBytes0, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&ctx.sendBuf1, sendBytes1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&ctx.recvBuf1, recvBytes1, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<uint8_t> hostSend0(sendBytes0);
    std::vector<uint8_t> hostSend1(sendBytes1);
    FillPattern(hostSend0, options.count0, options.dtype0, ctx.rank, 0);
    FillPattern(hostSend1, options.count1, options.dtype1, ctx.rank, 1);

    ACLCHECK(aclrtMemcpy(ctx.sendBuf0, sendBytes0, hostSend0.data(), sendBytes0, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(ctx.sendBuf1, sendBytes1, hostSend1.data(), sendBytes1, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(ctx.recvBuf0, recvBytes0, 0, recvBytes0));
    ACLCHECK(aclrtMemset(ctx.recvBuf1, recvBytes1, 0, recvBytes1));
    return 0;
}

static int VerifyResult(const RuntimeContext &ctx, const Options &options)
{
    size_t sendBytes0 = 0;
    size_t recvBytes0 = 0;
    size_t sendBytes1 = 0;
    size_t recvBytes1 = 0;
    if (!GetBytesByCount(options.count0, options.dtype0, sendBytes0) ||
        !GetBytesByCount(options.count1, options.dtype1, sendBytes1)) {
        return -1;
    }
    recvBytes0 = sendBytes0 * static_cast<size_t>(ctx.worldSize);
    recvBytes1 = sendBytes1 * static_cast<size_t>(ctx.worldSize);

    std::vector<uint8_t> hostRecv0(recvBytes0);
    std::vector<uint8_t> hostRecv1(recvBytes1);
    std::vector<uint8_t> expected0(recvBytes0);
    std::vector<uint8_t> expected1(recvBytes1);
    ACLCHECK(aclrtMemcpy(hostRecv0.data(), recvBytes0, ctx.recvBuf0, recvBytes0, ACL_MEMCPY_DEVICE_TO_HOST));
    ACLCHECK(aclrtMemcpy(hostRecv1.data(), recvBytes1, ctx.recvBuf1, recvBytes1, ACL_MEMCPY_DEVICE_TO_HOST));

    BuildExpected(expected0, options.count0, options.dtype0, ctx.worldSize, 0);
    BuildExpected(expected1, options.count1, options.dtype1, ctx.worldSize, 1);

    if (std::memcmp(hostRecv0.data(), expected0.data(), recvBytes0) != 0) {
        std::cerr << "Rank " << ctx.rank << " verify recvBuf0 failed" << std::endl;
        return -1;
    }
    if (std::memcmp(hostRecv1.data(), expected1.data(), recvBytes1) != 0) {
        std::cerr << "Rank " << ctx.rank << " verify recvBuf1 failed" << std::endl;
        return -1;
    }
    return 0;
}

static void Cleanup(RuntimeContext &ctx)
{
    if (ctx.comm != nullptr) {
        HcclCommDestroy(ctx.comm);
        ctx.comm = nullptr;
    }
    if (ctx.sendBuf0 != nullptr) {
        aclrtFree(ctx.sendBuf0);
        ctx.sendBuf0 = nullptr;
    }
    if (ctx.recvBuf0 != nullptr) {
        aclrtFree(ctx.recvBuf0);
        ctx.recvBuf0 = nullptr;
    }
    if (ctx.sendBuf1 != nullptr) {
        aclrtFree(ctx.sendBuf1);
        ctx.sendBuf1 = nullptr;
    }
    if (ctx.recvBuf1 != nullptr) {
        aclrtFree(ctx.recvBuf1);
        ctx.recvBuf1 = nullptr;
    }
    if (ctx.stream != nullptr) {
        aclrtDestroyStream(ctx.stream);
        ctx.stream = nullptr;
    }
    aclFinalize();
    MPI_Finalize();
}

static int RunBenchmark(const RuntimeContext &ctx, const Options &options, std::vector<double> &latenciesUs)
{
    for (int i = 0; i < options.warmup; ++i) {
        HCCLCHECK(HcclDoubleAllGatherCustom(ctx.sendBuf0, ctx.recvBuf0, options.count0, options.dtype0,
            ctx.sendBuf1, ctx.recvBuf1, options.count1, options.dtype1, ctx.comm, ctx.stream));
        ACLCHECK(aclrtSynchronizeStream(ctx.stream));
    }

    for (int i = 0; i < options.iters; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::steady_clock::now();
        HCCLCHECK(HcclDoubleAllGatherCustom(ctx.sendBuf0, ctx.recvBuf0, options.count0, options.dtype0,
            ctx.sendBuf1, ctx.recvBuf1, options.count1, options.dtype1, ctx.comm, ctx.stream));
        ACLCHECK(aclrtSynchronizeStream(ctx.stream));
        auto end = std::chrono::steady_clock::now();
        double localUs = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count();
        double globalUs = 0.0;
        MPI_Allreduce(&localUs, &globalUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        latenciesUs.push_back(globalUs);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        PrintUsage(argv[0]);
        return -1;
    }

    RuntimeContext ctx;
    int ret = InitEnv(argc, argv, ctx);
    if (ret != 0) {
        Cleanup(ctx);
        return ret;
    }

    ret = PrepareBuffers(ctx, options);
    if (ret == 0) {
        std::vector<double> latenciesUs;
        ret = RunBenchmark(ctx, options, latenciesUs);
        int verifyRet = 0;
        if (ret == 0) {
            verifyRet = VerifyResult(ctx, options);
        }
        int globalVerifyRet = 0;
        MPI_Allreduce(&verifyRet, &globalVerifyRet, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (ctx.rank == 0 && ret == 0 && !latenciesUs.empty()) {
            double minUs = *std::min_element(latenciesUs.begin(), latenciesUs.end());
            double maxUs = *std::max_element(latenciesUs.begin(), latenciesUs.end());
            double avgUs = std::accumulate(latenciesUs.begin(), latenciesUs.end(), 0.0) /
                static_cast<double>(latenciesUs.size());
            std::cout << "count0=" << options.count0
                      << " dtype0=" << DataTypeToString(options.dtype0)
                      << " count1=" << options.count1
                      << " dtype1=" << DataTypeToString(options.dtype1)
                      << " warmup=" << options.warmup
                      << " iters=" << options.iters << std::endl;
            std::cout << "latency(us): avg=" << avgUs
                      << " min=" << minUs
                      << " max=" << maxUs << std::endl;
            std::cout << "verify: " << (globalVerifyRet == 0 ? "PASS" : "FAIL") << std::endl;
        }
        ret = (ret == 0 && globalVerifyRet == 0) ? 0 : -1;
    }

    Cleanup(ctx);
    return ret;
}

