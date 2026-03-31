#include <cstdint>
#include <iostream>

#include "allgather_batch.h"

namespace {

alignas(32) uint8_t g_tokenInput[256] = {0};
alignas(32) uint8_t g_tokenOutput[256 * 8] = {0};
alignas(32) float g_scaleInput[8] = {0};
alignas(32) float g_scaleOutput[8 * 8] = {0};

}  // namespace

int main()
{
    // 这个 testcase 当前的目标是做“公开 API + 链接路径”的最小冒烟验证。
    // 这里故意传空 comm/stream，这样不依赖真实 runtime 也能验证我们已经走到 Host 参数校验层。
    HcclAllGatherItem items[2] = {};
    items[0].sendBuf = g_tokenInput;
    items[0].recvBuf = g_tokenOutput;
    items[0].sendCount = sizeof(g_tokenInput);
    items[0].dataType = HCCL_DATA_TYPE_INT8;

    items[1].sendBuf = g_scaleInput;
    items[1].recvBuf = g_scaleOutput;
    items[1].sendCount = sizeof(g_scaleInput) / sizeof(g_scaleInput[0]);
    items[1].dataType = HCCL_DATA_TYPE_FP32;

    HcclResult ret = HcclAllGatherBatch(items, 2, nullptr, nullptr);
    if (ret != HCCL_E_PTR) {
        std::cerr << "unexpected return code from smoke call: " << static_cast<int>(ret) << std::endl;
        return 1;
    }

    std::cout << "allgatherbatch testcase smoke passed: API is callable and parameter validation is reachable"
              << std::endl;
    return 0;
}
