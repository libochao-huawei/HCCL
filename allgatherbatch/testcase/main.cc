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

#include "allgather_batch.h"

namespace {

struct TestOptions {
    uint64_t tokenBytes = 320 * 1024;
    uint64_t scaleCount = 128;
    uint32_t requestedDeviceCount = 0;
    uint32_t printCount = 8;
    uint32_t warmupIters = 1;
    uint32_t measureIters = 1;
    bool onlyDeviceExecTime = false;
    bool verifyOutput = true;
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

void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " [--token-bytes N] [--scale-count N] [--devices N] [--print-count N]"
              << " [--warmup N] [--iters N] [--only-device-exec-time] [--no-verify]"
              << std::endl;
    std::cout << "  --token-bytes  int8 token bytes per rank, default 327680" << std::endl;
    std::cout << "  --scale-count  fp32 scale element count per rank, default 128" << std::endl;
    std::cout << "  --devices      number of devices to use, default all visible devices" << std::endl;
    std::cout << "  --print-count  number of values shown in preview, default 8" << std::endl;
    std::cout << "  --warmup       warmup iteration count before timing, default 1" << std::endl;
    std::cout << "  --iters        measured iteration count, default 1" << std::endl;
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

void PrintTokenPreview(const uint8_t *buf, uint64_t tokenBytes, uint32_t rankSize, uint32_t printCount, uint32_t device)
{
    const uint32_t limit = static_cast<uint32_t>(std::min<uint64_t>(tokenBytes, printCount));
    std::cout << "rank " << device << " token preview:";
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        std::cout << " [src " << rank << ":";
        const uint8_t *rankBase = buf + static_cast<size_t>(rank) * tokenBytes;
        for (uint32_t idx = 0; idx < limit; ++idx) {
            std::cout << " " << static_cast<uint32_t>(rankBase[idx]);
        }
        std::cout << " ]";
    }
    std::cout << std::endl;
}

void PrintScalePreview(const float *buf, uint64_t scaleCount, uint32_t rankSize, uint32_t printCount, uint32_t device)
{
    const uint32_t limit = static_cast<uint32_t>(std::min<uint64_t>(scaleCount, printCount));
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "rank " << device << " scale preview:";
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        std::cout << " [src " << rank << ":";
        const float *rankBase = buf + static_cast<size_t>(rank) * scaleCount;
        for (uint32_t idx = 0; idx < limit; ++idx) {
            std::cout << " " << rankBase[idx];
        }
        std::cout << " ]";
    }
    std::cout << std::endl;
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

    // 1. 参数初始化
    token.sendCount = ctx->options.tokenBytes;
    token.sendBytes = static_cast<size_t>(ctx->options.tokenBytes);
    token.recvBytes = static_cast<size_t>(ctx->options.tokenBytes * ctx->rankSize);
    token.dataType = HCCL_DATA_TYPE_INT8;

    scale.sendCount = ctx->options.scaleCount;
    scale.sendBytes = static_cast<size_t>(ctx->options.scaleCount * sizeof(float));
    scale.recvBytes = static_cast<size_t>(ctx->options.scaleCount * ctx->rankSize * sizeof(float));
    scale.dataType = HCCL_DATA_TYPE_FP32;

    // 2. 环境初始化与资源创建
    ACLCHECK_GOTO(aclrtSetDevice(static_cast<int32_t>(ctx->device)));
    HCCLCHECK_GOTO(HcclCommInitRootInfo(ctx->rankSize, ctx->rootInfo, ctx->device, &hcclComm));
    ACLCHECK_GOTO(aclrtCreateStream(&stream));

    // 创建计时和同步所需的 Event/Stream
    ACLCHECK_GOTO(aclrtCreateEvent(&startEvent));
    ACLCHECK_GOTO(aclrtCreateEvent(&endEvent));
    if (ctx->options.onlyDeviceExecTime) {
        ACLCHECK_GOTO(aclrtCreateStream(&syncStream));
        ACLCHECK_GOTO(aclrtCreateEventWithFlag(&syncEvent, ACL_EVENT_SYNC));
    }

    // 3. 内存分配与数据准备
    ACLCHECK_GOTO(aclrtMalloc(&token.sendDevice, token.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK_GOTO(aclrtMalloc(&token.recvDevice, token.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK_GOTO(aclrtMallocHost(&token.sendHost, token.sendBytes));
    ACLCHECK_GOTO(aclrtMallocHost(&token.recvHost, token.recvBytes));
    std::memset(token.recvHost, 0, token.recvBytes);
    FillTokenInput(static_cast<uint8_t *>(token.sendHost), ctx->options.tokenBytes, ctx->device);
    ACLCHECK_GOTO(
        aclrtMemcpy(token.sendDevice, token.sendBytes, token.sendHost, token.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    if (ctx->options.scaleCount > 0) {
        ACLCHECK_GOTO(aclrtMalloc(&scale.sendDevice, scale.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMalloc(&scale.recvDevice, scale.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMallocHost(&scale.sendHost, scale.sendBytes));
        ACLCHECK_GOTO(aclrtMallocHost(&scale.recvHost, scale.recvBytes));
        std::memset(scale.recvHost, 0, scale.recvBytes);
        FillScaleInput(static_cast<float *>(scale.sendHost), ctx->options.scaleCount, ctx->device);
        ACLCHECK_GOTO(
            aclrtMemcpy(scale.sendDevice, scale.sendBytes, scale.sendHost, scale.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // 4. 组装 Batch 任务
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

    // 5. 性能统计逻辑：Gate 阻塞（如果开启 onlyDeviceExecTime）
    if (ctx->options.onlyDeviceExecTime) {
        ACLCHECK_GOTO(aclrtStreamWaitEvent(stream, syncEvent));
        ACLCHECK_GOTO(aclrtResetEvent(syncEvent, stream));
    }

    // 6. Warmup
    for (uint32_t iter = 0; iter < ctx->options.warmupIters; ++iter) {
        HCCLCHECK_GOTO(HcclAllGatherBatch(items, itemCount, hcclComm, stream));
    }

    // 7. 正式测量执行
    ACLCHECK_GOTO(aclrtRecordEvent(startEvent, stream));
    for (uint32_t iter = 0; iter < ctx->options.measureIters; ++iter) {
        HCCLCHECK_GOTO(HcclAllGatherBatch(items, itemCount, hcclComm, stream));
    }
    ACLCHECK_GOTO(aclrtRecordEvent(endEvent, stream));
    if (ctx->options.onlyDeviceExecTime) {
        // 8. 触发执行（针对 onlyDeviceExecTime 模式）
        int sleepTime = 50 + ctx->options.warmupIters * 2 + ctx->options.measureIters * 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        ACLCHECK_GOTO(aclrtRecordEvent(syncEvent, syncStream));
    }
    // 9. 同步并计算时间与带宽
    ACLCHECK_GOTO(aclrtSynchronizeStream(stream));

    if (ctx->device == 0) {
        float ms = 0;
        ACLCHECK_GOTO(aclrtEventElapsedTime(&ms, startEvent, endEvent));
        
        // 统计所有参与计算的 buffer 总大小
        size_t totalBytes = token.sendBytes + (ctx->options.scaleCount > 0 ? scale.sendBytes : 0);
        unsigned long long dataSize = static_cast<unsigned long long>(totalBytes * ctx->rankSize);
        
        double total_time_us = static_cast<double>(ms * 1000.0f);
        double average_time_us = total_time_us / ctx->options.measureIters;
        
        constexpr double B_US_TO_GB_S = 1.0E6 / 1.0E9;
        double algorithm_bandwith_GBytes_s = static_cast<double>(dataSize) / average_time_us * B_US_TO_GB_S;

        printf("%-17llu | %-14.2f | %-20.5f | success \n", dataSize, average_time_us, algorithm_bandwith_GBytes_s);
    }

    ACLCHECK_GOTO(
        aclrtMemcpy(token.recvHost, token.recvBytes, token.recvDevice, token.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    // PrintTokenPreview(
    //     static_cast<const uint8_t *>(token.recvHost),
    //     ctx->options.tokenBytes,
    //     ctx->rankSize,
    //     ctx->options.printCount,
    //     ctx->device);

    if (ctx->options.verifyOutput &&
        !VerifyTokenOutput(static_cast<const uint8_t *>(token.recvHost), ctx->options.tokenBytes, ctx->rankSize)) {
        status = 1;
        goto CLEANUP;
    }

    {
        if (ctx->options.scaleCount > 0) {
            ACLCHECK_GOTO(
                aclrtMemcpy(scale.recvHost, scale.recvBytes, scale.recvDevice, scale.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
            // PrintScalePreview(
            //     static_cast<const float *>(scale.recvHost),
            //     ctx->options.scaleCount,
            //     ctx->rankSize,
            //     ctx->options.printCount,
            //     ctx->device);

            if (ctx->options.verifyOutput &&
                !VerifyScaleOutput(static_cast<const float *>(scale.recvHost), ctx->options.scaleCount, ctx->rankSize)) {
                status = 1;
                goto CLEANUP;
            }
        }
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
                  << ", onlyDeviceExecTime=" << (options.onlyDeviceExecTime ? "on" : "off")
                  << ", verify=" << (options.verifyOutput ? "on" : "off") << std::endl;

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
