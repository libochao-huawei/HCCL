#include <cstdlib>
#include <iostream>
#include <string>

#include "common.h"
#include "host_utils.h"

using namespace ops_hccl_allgather_batch;

namespace {

int ExpectTrue(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "[FAIL] " << message << std::endl;
        return 1;
    }
    return 0;
}

int TestBuildContextKey()
{
    const std::string key = BuildContextKey("demo_comm", 3);
    return ExpectTrue(key == "demo_comm_batch_3", "BuildContextKey should include comm name and item count");
}

int TestCheckItemValid()
{
    int send = 1;
    int recv[8] = {};
    HcclAllGatherItem validItem {&send, 4, HCCL_DATA_TYPE_INT32, recv};
    if (ExpectTrue(CheckItemValid(validItem, 0) == HCCL_SUCCESS, "valid item should pass") != 0) {
        return 1;
    }

    HcclAllGatherItem badCount {&send, 0, HCCL_DATA_TYPE_INT32, recv};
    if (ExpectTrue(CheckItemValid(badCount, 1) == HCCL_E_PARA, "zero sendCount should fail with HCCL_E_PARA") != 0) {
        return 1;
    }

    HcclAllGatherItem badType {&send, 1, HCCL_DATA_TYPE_RESERVED, recv};
    if (ExpectTrue(CheckItemValid(badType, 2) == HCCL_E_PARA, "unsupported data type should fail") != 0) {
        return 1;
    }

    HcclAllGatherItem nullSend {nullptr, 1, HCCL_DATA_TYPE_INT32, recv};
    if (ExpectTrue(CheckItemValid(nullSend, 3) == HCCL_E_PTR, "null sendBuf should fail with HCCL_E_PTR") != 0) {
        return 1;
    }

    HcclAllGatherItem nullRecv {&send, 1, HCCL_DATA_TYPE_INT32, nullptr};
    return ExpectTrue(CheckItemValid(nullRecv, 4) == HCCL_E_PTR, "null recvBuf should fail with HCCL_E_PTR");
}

int TestGetSliceSizeBytes()
{
    float send[2] = {0.0f, 1.0f};
    float recv[16] = {};
    HcclAllGatherItem item {send, 2, HCCL_DATA_TYPE_FP32, recv};
    return ExpectTrue(GetSliceSizeBytes(item) == sizeof(float) * 2, "slice size should be count * element size");
}

int TestDataTypeHelpers()
{
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_INT32) == sizeof(int32_t), "INT32 size mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_FP16) == sizeof(uint16_t), "FP16 size mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_RESERVED) == 0, "reserved type should map to 0") != 0) {
        return 1;
    }
    if (ExpectTrue(IsSupportedDataType(HCCL_DATA_TYPE_FP32), "FP32 should be supported") != 0) {
        return 1;
    }
    return ExpectTrue(!IsSupportedDataType(HCCL_DATA_TYPE_RESERVED), "reserved type should be unsupported");
}

} // namespace

int main()
{
    int failures = 0;
    failures += TestBuildContextKey();
    failures += TestCheckItemValid();
    failures += TestGetSliceSizeBytes();
    failures += TestDataTypeHelpers();

    if (failures != 0) {
        std::cerr << "[SUMMARY] host UT failed, count=" << failures << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[SUMMARY] host UT passed" << std::endl;
    return EXIT_SUCCESS;
}
