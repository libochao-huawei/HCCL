#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "hccl/hccl.h"
#include "allgather_batch.h"

namespace {

enum class BenchmarkMode {
    kCustom,
    kBaseline,
    kBoth,
};

struct TestOptions {
    uint64_t tokenBytes = 320 * 1024;
    uint64_t scaleCount = 128;
    uint32_t requestedDeviceCount = 0;
    uint32_t printCount = 8;
    uint32_t warmupIters = 1;
    uint32_t measureIters = 1;
    bool onlyDeviceExecTime = false;
    bool verifyOutput = true;
    BenchmarkMode mode = BenchmarkMode::kCustom;
};

struct BatchItemBuffer {
    void *sendDevice = nullptr;
    void *recvDevice = nullptr;
    void *sendHost = nullptr;
    void *recvHost = nullptr;
    uint64_t sendCount = 0;
    size_t sendBytes = 0;
    size_t recvBytes = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
};

struct ThreadContext {
    const HcclRootInfo *rootInfo = nullptr;
    uint32_t device = 0;
    uint32_t rankSize = 0;
    TestOptions options;
    std::atomic<int> *firstFailure = nullptr;
};

struct BenchmarkResult {
    unsigned long long dataSize = 0;
    double averageTimeUs = 0;
    double algorithmBandwidthGBps = 0;
};

#define ACLCHECK_GOTO(call)                                                                      \
    do {                                                                                         \
        aclError aclRet = (call);                                                                \
        if (aclRet != ACL_SUCCESS) {                                                             \
            std::cerr << "acl call failed at " << __FILE__ << ":" << __LINE__                  \
                      << ", ret=" << static_cast<int>(aclRet) << std::endl;                     \
            status = 1;                                                                          \
            goto CLEANUP;                                                                        \
        }                                                                                        \
    } while (0)

#define HCCLCHECK_GOTO(call)                                                                     \
    do {                                                                                         \
        HcclResult hcclRet = (call);                                                             \
        if (hcclRet != HCCL_SUCCESS) {                                                           \
            std::cerr << "hccl call failed at " << __FILE__ << ":" << __LINE__                 \
                      << ", ret=" << static_cast<int>(hcclRet) << std::endl;                    \
            status = 1;                                                                          \
            goto CLEANUP;                                                                        \
        }                                                                                        \
    } while (0)

const char *ToModeString(BenchmarkMode mode)
{
    switch (mode) {
        case BenchmarkMode::kCustom:
            return "custom";
        case BenchmarkMode::kBaseline:
            return "baseline";
        case BenchmarkMode::kBoth:
            return "both";
        default:
            return "unknown";
    }
}

bool ParseMode(const char *value, BenchmarkMode &mode)
{
    if (value == nullptr) {
        return false;
    }
    const std::string modeStr(value);
    if (modeStr == "custom") {
        mode = BenchmarkMode::kCustom;
        return true;
    }
    if (modeStr == "baseline") {
        mode = BenchmarkMode::kBaseline;
        return true;
    }
    if (modeStr == "both") {
        mode = BenchmarkMode::kBoth;
        return true;
    }
    return false;
}

void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " [--token-bytes N] [--scale-count N] [--devices N] [--print-count N]"
              << " [--warmup N] [--iters N] [--mode custom|baseline|both]"
              << " [--only-device-exec-time] [--no-verify]"
              << std::endl;
    std::cout << "  --token-bytes  int8 token bytes per rank, default 327680" << std::endl;
    std::cout << "  --scale-count  fp32 scale element count per rank, default 128" << std::endl;
    std::cout << "  --devices      number of devices to use, default all visible devices" << std::endl;
    std::cout << "  --print-count  number of values shown in preview, default 8" << std::endl;
    std::cout << "  --warmup       warmup iteration count before timing, default 1" << std::endl;
    std::cout << "  --iters        measured iteration count, default 1" << std::endl;
    std::cout << "  --mode         custom, baseline, or both; default custom" << std::endl;
    std::cout << "  --only-device-exec-time  gate the work queue so timing is closer to device execution" << std::endl;
    std::cout << "  --no-verify    skip host-side output verification" << std::endl;
}

bool ParseUint64(const char *value, uint64_t &result)
{
    if (value == nullptr || *value == '\0') {
        return false;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool ParseUint32(const char *value, uint32_t &result)
{
    uint64_t parsed = 0;
    if (!ParseUint64(value, parsed) || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    result = static_cast<uint32_t>(parsed);
    return true;
}

bool ParseArgs(int argc, char *argv[], TestOptions &options)
{
    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return false;
        }
        if (arg == "--no-verify") {
            options.verifyOutput = false;
            continue;
        }
        if (arg == "--only-device-exec-time") {
            options.onlyDeviceExecTime = true;
            continue;
        }
        if (idx + 1 >= argc) {
            std::cerr << "missing value for argument: " << arg << std::endl;
            return false;
        }

        if (arg == "--token-bytes") {
            if (!ParseUint64(argv[++idx], options.tokenBytes) || options.tokenBytes == 0) {
                std::cerr << "invalid token bytes" << std::endl;
                return false;
            }
        } else if (arg == "--scale-count") {
            if (!ParseUint64(argv[++idx], options.scaleCount)) {
                std::cerr << "invalid scale count" << std::endl;
                return false;
            }
        } else if (arg == "--devices") {
            if (!ParseUint32(argv[++idx], options.requestedDeviceCount) || options.requestedDeviceCount == 0) {
                std::cerr << "invalid device count" << std::endl;
                return false;
            }
        } else if (arg == "--print-count") {
            if (!ParseUint32(argv[++idx], options.printCount)) {
                std::cerr << "invalid print count" << std::endl;
                return false;
            }
        } else if (arg == "--warmup") {
            if (!ParseUint32(argv[++idx], options.warmupIters)) {
                std::cerr << "invalid warmup count" << std::endl;
                return false;
            }
        } else if (arg == "--iters") {
            if (!ParseUint32(argv[++idx], options.measureIters) || options.measureIters == 0) {
                std::cerr << "invalid measure iteration count" << std::endl;
                return false;
            }
        } else if (arg == "--mode") {
            if (!ParseMode(argv[++idx], options.mode)) {
                std::cerr << "invalid mode, expected custom|baseline|both" << std::endl;
                return false;
            }
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

void FillTokenInput(uint8_t *buf, uint64_t bytes, uint32_t rank)
{
    for (uint64_t idx = 0; idx < bytes; ++idx) {
        buf[idx] = static_cast<uint8_t>((rank + idx) & 0xffU);
    }
}

void FillScaleInput(float *buf, uint64_t count, uint32_t rank)
{
    for (uint64_t idx = 0; idx < count; ++idx) {
        buf[idx] = static_cast<float>(rank * 1000 + idx);
    }
}

bool VerifyTokenOutput(const uint8_t *buf, uint64_t tokenBytes, uint32_t rankSize)
{
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const uint8_t *rankBase = buf + static_cast<size_t>(rank) * tokenBytes;
        for (uint64_t idx = 0; idx < tokenBytes; ++idx) {
            const uint8_t expected = static_cast<uint8_t>((rank + idx) & 0xffU);
            if (rankBase[idx] != expected) {
                std::cerr << "token verify failed: srcRank=" << rank
                          << ", index=" << idx
                          << ", actual=" << static_cast<uint32_t>(rankBase[idx])
                          << ", expected=" << static_cast<uint32_t>(expected) << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool VerifyScaleOutput(const float *buf, uint64_t scaleCount, uint32_t rankSize)
{
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const float *rankBase = buf + static_cast<size_t>(rank) * scaleCount;
        for (uint64_t idx = 0; idx < scaleCount; ++idx) {
            const float expected = static_cast<float>(rank * 1000 + idx);
            if (rankBase[idx] != expected) {
                std::cerr << "scale verify failed: srcRank=" << rank
                          << ", index=" << idx
                          << ", actual=" << rankBase[idx]
                          << ", expected=" << expected << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool CopyAndVerifyOutputs(const ThreadContext &ctx, const BatchItemBuffer &token, const BatchItemBuffer &scale)
{
    if (!ctx.options.verifyOutput) {
        return true;
    }

    aclError aclRet = aclrtMemcpy(token.recvHost, token.recvBytes, token.recvDevice, token.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "acl call failed at " << __FILE__ << ":" << __LINE__
                  << ", ret=" << static_cast<int>(aclRet) << std::endl;
        return false;
    }
    if (!VerifyTokenOutput(static_cast<const uint8_t *>(token.recvHost), ctx.options.tokenBytes, ctx.rankSize)) {
        return false;
    }

    if (ctx.options.scaleCount > 0) {
        aclRet = aclrtMemcpy(scale.recvHost, scale.recvBytes, scale.recvDevice, scale.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "acl call failed at " << __FILE__ << ":" << __LINE__
                      << ", ret=" << static_cast<int>(aclRet) << std::endl;
            return false;
        }
        if (!VerifyScaleOutput(static_cast<const float *>(scale.recvHost), ctx.options.scaleCount, ctx.rankSize)) {
            return false;
        }
    }

    return true;
}

template <typename LaunchFn>
bool RunBenchmark(const char *modeLabel,
    const ThreadContext &ctx,
    aclrtStream stream,
    aclrtStream syncStream,
    aclrtEvent startEvent,
    aclrtEvent endEvent,
    aclrtEvent syncEvent,
    unsigned long long dataSize,
    LaunchFn launchFn,
    BenchmarkResult &result)
{
    auto checkAcl = [](aclError aclRet, const char *file, int line) -> bool {
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "acl call failed at " << file << ":" << line
                      << ", ret=" << static_cast<int>(aclRet) << std::endl;
            return false;
        }
        return true;
    };

    auto checkHccl = [](HcclResult hcclRet, const char *file, int line) -> bool {
        if (hcclRet != HCCL_SUCCESS) {
            std::cerr << "hccl call failed at " << file << ":" << line
                      << ", ret=" << static_cast<int>(hcclRet) << std::endl;
            return false;
        }
        return true;
    };

    if (ctx.options.onlyDeviceExecTime) {
        if (!checkAcl(aclrtStreamWaitEvent(stream, syncEvent), __FILE__, __LINE__)) {
            return false;
        }
        if (!checkAcl(aclrtResetEvent(syncEvent, stream), __FILE__, __LINE__)) {
            return false;
        }
    }

    for (uint32_t iter = 0; iter < ctx.options.warmupIters; ++iter) {
        if (!checkHccl(launchFn(), __FILE__, __LINE__)) {
            return false;
        }
    }

    if (!checkAcl(aclrtRecordEvent(startEvent, stream), __FILE__, __LINE__)) {
        return false;
    }
    for (uint32_t iter = 0; iter < ctx.options.measureIters; ++iter) {
        if (!checkHccl(launchFn(), __FILE__, __LINE__)) {
            return false;
        }
    }
    if (!checkAcl(aclrtRecordEvent(endEvent, stream), __FILE__, __LINE__)) {
        return false;
    }

    if (ctx.options.onlyDeviceExecTime) {
        const int sleepTime = 50 + static_cast<int>(ctx.options.warmupIters) * 2 + static_cast<int>(ctx.options.measureIters) * 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        if (!checkAcl(aclrtRecordEvent(syncEvent, syncStream), __FILE__, __LINE__)) {
            return false;
        }
    }

    if (!checkAcl(aclrtSynchronizeStream(stream), __FILE__, __LINE__)) {
        return false;
    }

    float ms = 0;
    if (!checkAcl(aclrtEventElapsedTime(&ms, startEvent, endEvent), __FILE__, __LINE__)) {
        return false;
    }

    result.dataSize = dataSize;
    result.averageTimeUs = static_cast<double>(ms * 1000.0f) / ctx.options.measureIters;
    constexpr double B_US_TO_GB_S = 1.0E6 / 1.0E9;
    result.algorithmBandwidthGBps = static_cast<double>(dataSize) / result.averageTimeUs * B_US_TO_GB_S;

    if (ctx.device == 0) {
        printf("%-10s | %-17llu | %-14.2f | %-20.5f | success \n",
            modeLabel,
            result.dataSize,
            result.averageTimeUs,
            result.algorithmBandwidthGBps);
    }
    return true;
}

int RunOnDevice(void *arg)
{
    ThreadContext *ctx = static_cast<ThreadContext *>(arg);
    int status = 0;

    HcclComm hcclComm = nullptr;
    aclrtStream stream = nullptr;
    aclrtStream syncStream = nullptr;
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    aclrtEvent syncEvent = nullptr;

    BatchItemBuffer token;
    BatchItemBuffer scale;
    HcclAllGatherItem items[2] = {};
    uint32_t itemCount = 0;
    BenchmarkResult customResult = {};
    BenchmarkResult baselineResult = {};
    size_t totalBytes = 0;
    unsigned long long dataSize = 0;

    auto launchCustom = [&]() -> HcclResult {
        return HcclAllGatherBatch(items, itemCount, hcclComm, stream);
    };

    auto launchBaseline = [&]() -> HcclResult {
        HcclResult ret = HcclAllGather(token.sendDevice, token.recvDevice, token.sendCount, token.dataType, hcclComm, stream);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        if (ctx->options.scaleCount > 0) {
            ret = HcclAllGather(scale.sendDevice, scale.recvDevice, scale.sendCount, scale.dataType, hcclComm, stream);
        }
        return ret;
    };

    token.sendCount = ctx->options.tokenBytes;
    token.sendBytes = static_cast<size_t>(ctx->options.tokenBytes);
    token.recvBytes = static_cast<size_t>(ctx->options.tokenBytes * ctx->rankSize);
    token.dataType = HCCL_DATA_TYPE_INT8;

    scale.sendCount = ctx->options.scaleCount;
    scale.sendBytes = static_cast<size_t>(ctx->options.scaleCount * sizeof(float));
    scale.recvBytes = static_cast<size_t>(ctx->options.scaleCount * ctx->rankSize * sizeof(float));
    scale.dataType = HCCL_DATA_TYPE_FP32;

    ACLCHECK_GOTO(aclrtSetDevice(static_cast<int32_t>(ctx->device)));
    HCCLCHECK_GOTO(HcclCommInitRootInfo(ctx->rankSize, ctx->rootInfo, ctx->device, &hcclComm));
    ACLCHECK_GOTO(aclrtCreateStream(&stream));
    ACLCHECK_GOTO(aclrtCreateEvent(&startEvent));
    ACLCHECK_GOTO(aclrtCreateEvent(&endEvent));
    if (ctx->options.onlyDeviceExecTime) {
        ACLCHECK_GOTO(aclrtCreateStream(&syncStream));
        ACLCHECK_GOTO(aclrtCreateEventWithFlag(&syncEvent, ACL_EVENT_SYNC));
    }

    ACLCHECK_GOTO(aclrtMalloc(&token.sendDevice, token.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK_GOTO(aclrtMalloc(&token.recvDevice, token.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK_GOTO(aclrtMallocHost(&token.sendHost, token.sendBytes));
    ACLCHECK_GOTO(aclrtMallocHost(&token.recvHost, token.recvBytes));
    std::memset(token.recvHost, 0, token.recvBytes);
    FillTokenInput(static_cast<uint8_t *>(token.sendHost), ctx->options.tokenBytes, ctx->device);
    ACLCHECK_GOTO(aclrtMemcpy(token.sendDevice, token.sendBytes, token.sendHost, token.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    if (ctx->options.scaleCount > 0) {
        ACLCHECK_GOTO(aclrtMalloc(&scale.sendDevice, scale.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMalloc(&scale.recvDevice, scale.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMallocHost(&scale.sendHost, scale.sendBytes));
        ACLCHECK_GOTO(aclrtMallocHost(&scale.recvHost, scale.recvBytes));
        std::memset(scale.recvHost, 0, scale.recvBytes);
        FillScaleInput(static_cast<float *>(scale.sendHost), ctx->options.scaleCount, ctx->device);
        ACLCHECK_GOTO(aclrtMemcpy(scale.sendDevice, scale.sendBytes, scale.sendHost, scale.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    items[itemCount].sendBuf = token.sendDevice;
    items[itemCount].recvBuf = token.recvDevice;
    items[itemCount].sendCount = token.sendCount;
    items[itemCount].dataType = token.dataType;
    ++itemCount;
    if (ctx->options.scaleCount > 0) {
        items[itemCount].sendBuf = scale.sendDevice;
        items[itemCount].recvBuf = scale.recvDevice;
        items[itemCount].sendCount = scale.sendCount;
        items[itemCount].dataType = scale.dataType;
        ++itemCount;
    }

    totalBytes = token.sendBytes + (ctx->options.scaleCount > 0 ? scale.sendBytes : 0);
    dataSize = static_cast<unsigned long long>(totalBytes * ctx->rankSize);

    if (ctx->options.mode == BenchmarkMode::kCustom || ctx->options.mode == BenchmarkMode::kBoth) {
        if (!RunBenchmark("custom",
                *ctx,
                stream,
                syncStream,
                startEvent,
                endEvent,
                syncEvent,
                dataSize,
                launchCustom,
                customResult)) {
            status = 1;
            goto CLEANUP;
        }
        if (!CopyAndVerifyOutputs(*ctx, token, scale)) {
            status = 1;
            goto CLEANUP;
        }
    }

    if (ctx->options.mode == BenchmarkMode::kBaseline || ctx->options.mode == BenchmarkMode::kBoth) {
        if (!RunBenchmark("baseline",
                *ctx,
                stream,
                syncStream,
                startEvent,
                endEvent,
                syncEvent,
                dataSize,
                launchBaseline,
                baselineResult)) {
            status = 1;
            goto CLEANUP;
        }
        if (!CopyAndVerifyOutputs(*ctx, token, scale)) {
            status = 1;
            goto CLEANUP;
        }
    }

    if (ctx->device == 0 && ctx->options.mode == BenchmarkMode::kBoth) {
        const double deltaUs = baselineResult.averageTimeUs - customResult.averageTimeUs;
        const double speedup = (customResult.averageTimeUs > 0.0) ?
            (baselineResult.averageTimeUs / customResult.averageTimeUs) : 0.0;
        std::cout << "compare    | delta(us)=" << std::fixed << std::setprecision(2) << deltaUs
                  << ", speedup=" << speedup << "x" << std::endl;
    }

CLEANUP:
    if (startEvent != nullptr) {
        (void)aclrtDestroyEvent(startEvent);
    }
    if (endEvent != nullptr) {
        (void)aclrtDestroyEvent(endEvent);
    }
    if (syncEvent != nullptr) {
        (void)aclrtDestroyEvent(syncEvent);
    }
    if (syncStream != nullptr) {
        (void)aclrtDestroyStream(syncStream);
    }
    if (hcclComm != nullptr) {
        (void)HcclCommDestroy(hcclComm);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (token.sendDevice != nullptr) {
        (void)aclrtFree(token.sendDevice);
    }
    if (token.recvDevice != nullptr) {
        (void)aclrtFree(token.recvDevice);
    }
    if (token.sendHost != nullptr) {
        (void)aclrtFreeHost(token.sendHost);
    }
    if (token.recvHost != nullptr) {
        (void)aclrtFreeHost(token.recvHost);
    }
    if (scale.sendDevice != nullptr) {
        (void)aclrtFree(scale.sendDevice);
    }
    if (scale.recvDevice != nullptr) {
        (void)aclrtFree(scale.recvDevice);
    }
    if (scale.sendHost != nullptr) {
        (void)aclrtFreeHost(scale.sendHost);
    }
    if (scale.recvHost != nullptr) {
        (void)aclrtFreeHost(scale.recvHost);
    }
    (void)aclrtResetDevice(ctx->device);

    if (status != 0 && ctx->firstFailure != nullptr) {
        int expected = 0;
        (void)ctx->firstFailure->compare_exchange_strong(expected, status);
    }
    return status;
}

}  // namespace

int main(int argc, char *argv[])
{
    TestOptions options;
    if (!ParseArgs(argc, argv, options)) {
        return 1;
    }

    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "aclInit failed, ret=" << static_cast<int>(aclRet) << std::endl;
        return 1;
    }

    int status = 0;
    HcclRootInfo *rootInfo = nullptr;
    uint32_t deviceCount = 0;

    do {
        aclRet = aclrtGetDeviceCount(&deviceCount);
        if (aclRet != ACL_SUCCESS || deviceCount == 0) {
            std::cerr << "aclrtGetDeviceCount failed or no device found, ret=" << static_cast<int>(aclRet)
                      << ", deviceCount=" << deviceCount << std::endl;
            status = 1;
            break;
        }

        const uint32_t usedDeviceCount =
            (options.requestedDeviceCount == 0) ? deviceCount : std::min(options.requestedDeviceCount, deviceCount);
        std::cout << "allgatherbatch testcase starts with devices=" << usedDeviceCount
                  << ", tokenBytes=" << options.tokenBytes
                  << ", scaleCount=" << options.scaleCount
                  << ", warmup=" << options.warmupIters
                  << ", iters=" << options.measureIters
                  << ", mode=" << ToModeString(options.mode)
                  << ", onlyDeviceExecTime=" << (options.onlyDeviceExecTime ? "on" : "off")
                  << ", verify=" << (options.verifyOutput ? "on" : "off") << std::endl;
        std::cout << "mode       | dataSize(B)        | avgTime(us)    | algoBandwidth(GB/s)  | status" << std::endl;

        aclRet = aclrtSetDevice(0);
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "aclrtSetDevice(0) failed, ret=" << static_cast<int>(aclRet) << std::endl;
            status = 1;
            break;
        }

        aclRet = aclrtMallocHost(reinterpret_cast<void **>(&rootInfo), sizeof(HcclRootInfo));
        if (aclRet != ACL_SUCCESS || rootInfo == nullptr) {
            std::cerr << "aclrtMallocHost(rootInfo) failed, ret=" << static_cast<int>(aclRet) << std::endl;
            status = 1;
            break;
        }

        HcclResult hcclRet = HcclGetRootInfo(rootInfo);
        if (hcclRet != HCCL_SUCCESS) {
            std::cerr << "HcclGetRootInfo failed, ret=" << static_cast<int>(hcclRet) << std::endl;
            status = 1;
            break;
        }

        std::atomic<int> firstFailure(0);
        std::vector<std::thread> threads(usedDeviceCount);
        std::vector<ThreadContext> contexts(usedDeviceCount);
        for (uint32_t rank = 0; rank < usedDeviceCount; ++rank) {
            contexts[rank].rootInfo = rootInfo;
            contexts[rank].device = rank;
            contexts[rank].rankSize = usedDeviceCount;
            contexts[rank].options = options;
            contexts[rank].firstFailure = &firstFailure;
            threads[rank] = std::thread(RunOnDevice, static_cast<void *>(&contexts[rank]));
        }

        for (uint32_t rank = 0; rank < usedDeviceCount; ++rank) {
            threads[rank].join();
        }
        status = firstFailure.load();
    } while (false);

    if (rootInfo != nullptr) {
        (void)aclrtFreeHost(rootInfo);
    }
    (void)aclrtResetDevice(0);
    (void)aclFinalize();

    if (status != 0) {
        std::cerr << "allgatherbatch testcase failed" << std::endl;
        return status;
    }

    std::cout << "allgatherbatch testcase finished successfully" << std::endl;
    return 0;
}
