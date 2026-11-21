/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "aicpu_stuck_counter.h"
#include "stuck_stub.h"

namespace {

using namespace ops_hccl;

class AicpuStuckCounterTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ut_stub::g_reduceCalls.clear();
        ut_stub::g_reduceRet = 0;
        ut_stub::g_reduceFailTimes = 0;
    }
};

TEST_F(AicpuStuckCounterTest, EmitSkipWhenDisabled)
{
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 0;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());
}

TEST_F(AicpuStuckCounterTest, EmitSkipWhenThreadZero)
{
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    EXPECT_FALSE(AicpuStuckCounterHead(0, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());
}

TEST_F(AicpuStuckCounterTest, EmitSkipsBothWhenAnyAddrZero)
{
    OpParam param;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;

    // head=0、tail 非 0：两侧都不发，避免 head<tail 永久失衡
    param.counterHeadAddr = 0;
    param.counterTailAddr = 0x20;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());

    // head 非 0、tail=0：两侧都不发，避免 head>tail 永久失衡
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());

    // addOne=0：两侧都不发
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());
}

TEST_F(AicpuStuckCounterTest, EmitHeadUsesHeadAddrTailUsesTailAddr)
{
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    EXPECT_TRUE(AicpuStuckCounterHead(0x7, param));
    EXPECT_TRUE(AicpuStuckCounterTail(0x7, param));
    ASSERT_EQ(ut_stub::g_reduceCalls.size(), 2u);

    const auto& head = ut_stub::g_reduceCalls[0];
    EXPECT_EQ(head.thread, 0x7u);
    EXPECT_EQ(head.dst, 0x10u);
    EXPECT_EQ(head.src, 0x30u);
    EXPECT_EQ(head.count, 1u);
    EXPECT_EQ(head.dataType, static_cast<int>(HCOMM_DATA_TYPE_INT32));
    EXPECT_EQ(head.reduceOp, static_cast<int>(HCOMM_REDUCE_SUM));

    const auto& tail = ut_stub::g_reduceCalls[1];
    EXPECT_EQ(tail.dst, 0x20u);
    EXPECT_EQ(tail.src, 0x30u);
}

TEST_F(AicpuStuckCounterTest, EmitReturnsFalseOnReduceFail)
{
    ut_stub::g_reduceRet = HCCL_E_INTERNAL;
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    // 失败时返回 false，调用方据此跳过 tail，避免 head/tail 计数器单向偏移
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_EQ(ut_stub::g_reduceCalls.size(), 1u);
}

TEST_F(AicpuStuckCounterTest, TailRetriesOnceAfterReduceFail)
{
    ut_stub::g_reduceFailTimes = 1;
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    // 首次失败后重试一次成功：仅 tail 重试，不会双计数
    EXPECT_TRUE(AicpuStuckCounterTail(0x1, param));
    ASSERT_EQ(ut_stub::g_reduceCalls.size(), 2u);
    EXPECT_EQ(ut_stub::g_reduceCalls[0].dst, 0x20u);
    EXPECT_EQ(ut_stub::g_reduceCalls[1].dst, 0x20u);
}

TEST_F(AicpuStuckCounterTest, TailReturnsFalseWhenRetryAlsoFails)
{
    ut_stub::g_reduceFailTimes = 2;
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    // 两次都失败：返回 false，由调用方按“未发射”处理（cache 录制期作废 tag）
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_EQ(ut_stub::g_reduceCalls.size(), 2u);
}

TEST_F(AicpuStuckCounterTest, HeadDoesNotRetryAfterReduceFail)
{
    ut_stub::g_reduceFailTimes = 1;
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    // head 保持单次发射，避免双计数
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_EQ(ut_stub::g_reduceCalls.size(), 1u);
}

TEST_F(AicpuStuckCounterTest, TailSkipsRetryWhenNotReady)
{
    OpParam param;
    param.isEnableCounter = 0;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    // 未使能时不重试，也不产生调用
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());
}

TEST_F(AicpuStuckCounterTest, CacheInvalidateOnlyWhenCounterEnabledAndMarkerMissing)
{
    OpParam param;
    param.isEnableCounter = 1;
    EXPECT_FALSE(AicpuStuckCounterNeedInvalidateCache(param, true, true));
    EXPECT_TRUE(AicpuStuckCounterNeedInvalidateCache(param, false, true));
    EXPECT_TRUE(AicpuStuckCounterNeedInvalidateCache(param, true, false));
    EXPECT_TRUE(AicpuStuckCounterNeedInvalidateCache(param, false, false));

    // 计数未使能时缓存不含计数 SQE，无需作废
    param.isEnableCounter = 0;
    EXPECT_FALSE(AicpuStuckCounterNeedInvalidateCache(param, false, false));
}

} // namespace
