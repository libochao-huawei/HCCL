/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ALLGATHER_RING_TEMPLATE_H
#define ALLGATHER_RING_TEMPLATE_H

#include "alg_template_base.h"

namespace ops_hccl {
// Ring 模板算法：
// 负责具体的点到点收发时序，不关心外层资源申请。
class AllGatherRing : public AlgTemplateBase {
public:
    explicit AllGatherRing();
    ~AllGatherRing() override;

    // 异步执行 ring 通信主流程。
    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels) override;

private:
    // 先把本 rank 输入拷贝到输出中属于自己的切片位置。
    HcclResult PrepareLocalSlice();

    // 当前 rank 编号（在本层通信域内）。
    u32 interRank_ = 0;
    // 本层通信域 rank 数。
    u32 interRankSize_ = 0;
    // 左邻居通道（接收方向）。
    ChannelInfo channelLeft_;
    // 右邻居通道（发送方向）。
    ChannelInfo channelRight_;
};
}

#endif
