/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include "gtest/gtest.h"
#include "op_common.h"

using namespace ops_hccl;

namespace {

struct LocalTagKey128 {
    u64 lo {0};
    u64 hi {0};

    bool operator==(const LocalTagKey128 &other) const noexcept
    {
        return lo == other.lo && hi == other.hi;
    }
};

struct LocalTagKey128Hasher {
    size_t operator()(const LocalTagKey128 &key) const noexcept
    {
        u64 x = key.lo ^ (key.hi + 0x9e3779b97f4a7c15ULL + (key.lo << 6) + (key.lo >> 2));
        return static_cast<size_t>(x);
    }
};

struct LocalFastLaunchTlsEntry {
    bool valid {false};
    LocalTagKey128 key {};
    HcclComm comm {nullptr};
    void *ctx {nullptr};
};

constexpr size_t LOCAL_BUCKET_NUM = 8;
thread_local std::array<LocalFastLaunchTlsEntry, LOCAL_BUCKET_NUM> g_localL1Cache;
thread_local std::unordered_map<LocalTagKey128, void*, LocalTagKey128Hasher> g_localL2Cache;

size_t LocalBucket(const LocalTagKey128 &key)
{
    return static_cast<size_t>((key.lo ^ key.hi) & (LOCAL_BUCKET_NUM - 1));
}

bool LocalTryL1(const LocalTagKey128 &key, HcclComm comm, void **ctx)
{
    auto &entry = g_localL1Cache[LocalBucket(key)];
    if (entry.valid && entry.comm == comm && entry.key == key && entry.ctx != nullptr) {
        *ctx = entry.ctx;
        return true;
    }
    return false;
}

bool LocalTryL2(const LocalTagKey128 &key, void **ctx)
{
    auto it = g_localL2Cache.find(key);
    if (it == g_localL2Cache.end() || it->second == nullptr) {
        return false;
    }
    *ctx = it->second;
    return true;
}

void LocalPutL1(const LocalTagKey128 &key, HcclComm comm, void *ctx)
{
    auto &entry = g_localL1Cache[LocalBucket(key)];
    entry.valid = true;
    entry.key = key;
    entry.comm = comm;
    entry.ctx = ctx;
}

void LocalPutL2(const LocalTagKey128 &key, void *ctx)
{
    g_localL2Cache[key] = ctx;
}

class FastLaunchCacheUt : public testing::Test {
protected:
    void SetUp() override
    {
        for (auto &entry : g_localL1Cache) {
            entry = LocalFastLaunchTlsEntry{};
        }
        g_localL2Cache.clear();
    }

    OpParam MakeBaseParam(const char *tag, HcclCMDType opType, HcclDataType dataType, u64 count,
                          HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED,
                          u32 root = INVALID_VALUE_RANKID)
    {
        OpParam param{};
        std::memset(&param, 0, sizeof(param));
        std::snprintf(param.tag, sizeof(param.tag), "%s", tag);
        param.opType = opType;
        param.DataDes.count = count;
        param.DataDes.dataType = dataType;
        param.DataDes.outputType = HCCL_DATA_TYPE_RESERVED;
        param.reduceType = reduceType;
        param.root = root;
        return param;
    }
};

TEST_F(FastLaunchCacheUt, set_fast_launch_tag_allreduce_formats_expected_value)
{
    OpParam param = MakeBaseParam("demo_tag", HcclCMDType::HCCL_CMD_ALLREDUCE,
                                  HcclDataType::HCCL_DATA_TYPE_FP16, 1024,
                                  HcclReduceOp::HCCL_REDUCE_SUM);

    ASSERT_EQ(SetOpParamFastLaunchTag(param), HCCL_SUCCESS);
    EXPECT_STREQ(param.fastLaunchTag, "demo_tag_float16_sum_1024");
}

TEST_F(FastLaunchCacheUt, set_fast_launch_tag_broadcast_adds_root_suffix)
{
    OpParam param = MakeBaseParam("broadcast_tag", HcclCMDType::HCCL_CMD_BROADCAST,
                                  HcclDataType::HCCL_DATA_TYPE_INT32, 16,
                                  HcclReduceOp::HCCL_REDUCE_RESERVED, 3);

    ASSERT_EQ(SetOpParamFastLaunchTag(param), HCCL_SUCCESS);
    EXPECT_STREQ(param.fastLaunchTag, "broadcast_tag_int32_16_r3");
}

TEST_F(FastLaunchCacheUt, set_fast_launch_tag_alltoallv_skips_count_field)
{
    OpParam param{};
    std::memset(&param, 0, sizeof(param));
    std::snprintf(param.tag, sizeof(param.tag), "%s", "a2a_v_tag");
    param.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    param.all2AllVDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    param.DataDes.count = 4096; // alltoallv fastLaunchTag 不应使用它

    ASSERT_EQ(SetOpParamFastLaunchTag(param), HCCL_SUCCESS);
    EXPECT_STREQ(param.fastLaunchTag, "a2a_v_tag_uint8");
}

TEST_F(FastLaunchCacheUt, local_l2_cache_recovers_entry_after_l1_collision)
{
    const HcclComm fakeComm = reinterpret_cast<HcclComm>(0x1234);
    LocalTagKey128 first {1, 1};
    LocalTagKey128 second {9, 1}; // 与 first 命中同一个 bucket

    void *ctx1 = reinterpret_cast<void *>(0x1000);
    void *ctx2 = reinterpret_cast<void *>(0x2000);
    void *queryCtx = nullptr;

    LocalPutL1(first, fakeComm, ctx1);
    LocalPutL2(first, ctx1);

    ASSERT_TRUE(LocalTryL1(first, fakeComm, &queryCtx));
    EXPECT_EQ(queryCtx, ctx1);

    // 同 bucket 写入，模拟 L1 被覆盖
    LocalPutL1(second, fakeComm, ctx2);
    LocalPutL2(second, ctx2);

    queryCtx = nullptr;
    EXPECT_FALSE(LocalTryL1(first, fakeComm, &queryCtx));
    EXPECT_TRUE(LocalTryL2(first, &queryCtx));
    EXPECT_EQ(queryCtx, ctx1);
}

} // namespace

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
