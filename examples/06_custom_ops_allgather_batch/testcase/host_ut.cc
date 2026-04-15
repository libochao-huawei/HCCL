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
    const std::string key = BuildContextKey("AllGatherBatch_demo_Custom", 0x1234, 3);
    if (ExpectTrue(key == "AllGatherBatch_demo_Custom_comm_4660_n3",
                   "BuildContextKey should include tag, comm identity and item count") != 0) {
        return 1;
    }
    const std::string engineCtxTag = BuildEngineCtxTag("AllGatherBatch_demo_Custom", 3);
    return ExpectTrue(engineCtxTag == "AllGatherBatch_demo_Custom_n3",
                      "BuildEngineCtxTag should include tag and item count");
}

int TestCheckItemValid()
{
    int send = 1;
    int recv[8] = {};
    HcclAllGatherItem validItem {&send, 4, HCCL_DATA_TYPE_INT32, recv};
    if (ExpectTrue(CheckItemValid(validItem, 0, 1, 8) == HCCL_SUCCESS, "valid item should pass") != 0) {
        return 1;
    }

    HcclAllGatherItem badCount {&send, 0, HCCL_DATA_TYPE_INT32, recv};
    if (ExpectTrue(CheckItemValid(badCount, 1, 1, 8) == HCCL_E_PARA,
                   "zero sendCount should fail with HCCL_E_PARA") != 0) {
        return 1;
    }

    HcclAllGatherItem badType {&send, 1, HCCL_DATA_TYPE_RESERVED, recv};
    if (ExpectTrue(CheckItemValid(badType, 2, 1, 8) == HCCL_E_PARA, "unsupported data type should fail") != 0) {
        return 1;
    }

    HcclAllGatherItem nullSend {nullptr, 1, HCCL_DATA_TYPE_INT32, recv};
    if (ExpectTrue(CheckItemValid(nullSend, 3, 1, 8) == HCCL_E_PTR, "null sendBuf should fail with HCCL_E_PTR") != 0) {
        return 1;
    }

    HcclAllGatherItem nullRecv {&send, 1, HCCL_DATA_TYPE_INT32, nullptr};
    if (ExpectTrue(CheckItemValid(nullRecv, 4, 1, 8) == HCCL_E_PTR, "null recvBuf should fail with HCCL_E_PTR") != 0) {
        return 1;
    }

    HcclAllGatherItem overflowOffset {&send, UINT64_MAX / sizeof(int32_t), HCCL_DATA_TYPE_INT32, recv};
    return ExpectTrue(CheckItemValid(overflowOffset, 5, 2, 8) == HCCL_E_PARA,
                      "offset overflow should fail with HCCL_E_PARA");
}

int TestGetSliceSizeBytes()
{
    float send[2] = {0.0f, 1.0f};
    float recv[16] = {};
    HcclAllGatherItem item {send, 2, HCCL_DATA_TYPE_FP32, recv};
    return ExpectTrue(GetSliceSizeBytes(item) == sizeof(float) * 2, "slice size should be count * element size");
}

int TestPackBatchItemsForLaunch()
{
    int32_t send0[4] = {0, 1, 2, 3};
    int32_t recv0[16] = {};
    float send1[2] = {1.0f, 2.0f};
    float recv1[8] = {};

    OpParam param;
    param.rank = 2;
    param.rankSize = 4;
    param.itemCount = 2;
    param.items[0] = HcclAllGatherItem{send0, 4, HCCL_DATA_TYPE_INT32, recv0};
    param.items[1] = HcclAllGatherItem{send1, 2, HCCL_DATA_TYPE_FP32, recv1};

    PackedBatchItem packed[MAX_ITEM_COUNT] = {};
    PackBatchItemsForLaunch(param, packed, MAX_ITEM_COUNT);

    if (ExpectTrue(packed[0].inputAddr == reinterpret_cast<uint64_t>(send0), "packed inputAddr[0] mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(packed[0].outputAddr == reinterpret_cast<uint64_t>(recv0), "packed outputAddr[0] mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(packed[0].sliceSize == sizeof(int32_t) * 4, "packed sliceSize[0] mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(packed[0].offset == static_cast<uint64_t>(param.rank) * packed[0].sliceSize,
                   "packed offset[0] mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(packed[0].token != 0, "packed token[0] should not be zero") != 0) {
        return 1;
    }

    if (ExpectTrue(packed[1].inputAddr == reinterpret_cast<uint64_t>(send1), "packed inputAddr[1] mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(packed[1].outputAddr == reinterpret_cast<uint64_t>(recv1), "packed outputAddr[1] mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(packed[1].sliceSize == sizeof(float) * 2, "packed sliceSize[1] mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(packed[1].offset == static_cast<uint64_t>(param.rank) * packed[1].sliceSize,
                   "packed offset[1] mismatch") != 0) {
        return 1;
    }
    return ExpectTrue(packed[1].token != 0, "packed token[1] should not be zero");
}

int TestDataTypeHelpers()
{
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_INT32) == sizeof(int32_t), "INT32 size mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_UINT64) == sizeof(uint64_t), "UINT64 size mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_FP16) == sizeof(uint16_t), "FP16 size mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_BFP16) == sizeof(uint16_t), "BFP16 size mismatch") != 0) {
        return 1;
    }
    if (ExpectTrue(GetDataTypeSize(HCCL_DATA_TYPE_RESERVED) == 0, "reserved type should map to 0") != 0) {
        return 1;
    }
    if (ExpectTrue(IsSupportedDataType(HCCL_DATA_TYPE_FP32), "FP32 should be supported") != 0) {
        return 1;
    }
    if (ExpectTrue(IsSupportedDataType(HCCL_DATA_TYPE_UINT64), "UINT64 should be supported") != 0) {
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
    failures += TestPackBatchItemsForLaunch();
    failures += TestDataTypeHelpers();

    if (failures != 0) {
        std::cerr << "[SUMMARY] host UT failed, count=" << failures << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[SUMMARY] host UT passed" << std::endl;
    return EXIT_SUCCESS;
}
