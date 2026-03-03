/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "alg_template_register.h"
#include "allgather_ring.h"

namespace ops_hccl {
AllGatherRing::AllGatherRing() : AlgTemplateBase()
{
    // 构造函数保持轻量，具体资源由 Prepare/RunAsync 阶段注入。
}

AllGatherRing::~AllGatherRing()
{
    // 模板对象无自有堆资源，默认析构即可。
}

HcclResult AllGatherRing::PrepareLocalSlice()
{
    // 本 rank 的数据最终应该落到 slices_[interRank_] 对应位置。
    const Slice &selfSlice = slices_[interRank_];
    void *src = inputMem_.addr;
    void *dst = static_cast<u8 *>(outputMem_.addr) + selfSlice.offset;
    // 本地拷贝：先把自己的分片放到输出缓冲，后续 ring 交换补齐其他分片。
    return static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, dst, src, selfSlice.size));
}

HcclResult AllGatherRing::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    // 运行前校验输入输出地址有效。
    CHK_PTR_NULL(inputMem_.addr);
    CHK_PTR_NULL(outputMem_.addr);

    // 缓存 rank 信息。
    interRank_ = rank;
    interRankSize_ = rankSize;
    // slices_ 由外层 Prepare 传入，数量应与 rankSize 一致。
    CHK_PRT_RET((slices_.size() != interRankSize_), HCCL_ERROR("[AllGatherRing][RunAsync] invalid slice size[%llu], "
        "rankSize[%u]", slices_.size(), interRankSize_), HCCL_E_PARA);
    // 至少需要每个 rank 一个通道描述。
    CHK_PRT_RET((channels.size() < rankSize), HCCL_ERROR("[AllGatherRing][RunAsync] link size[%llu] is less than "
        "rank size[%u]", channels.size(), rankSize), HCCL_E_INTERNAL);

    // 构建环上的左右邻居。
    const u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    const u32 ringNextRank = (rank + 1) % rankSize;
    channelLeft_ = channels[ringPrevRank];
    channelRight_ = channels[ringNextRank];

    // 第 0 步：先把本地数据放进输出的自有分片。
    CHK_RET(PrepareLocalSlice());
    // sendIdx 表示当前轮要发送哪个分片索引。
    u32 sendIdx = interRank_;
    // 标准 ring allgather 需要 rankSize-1 轮交换，最终补齐其余分片。
    for (u32 step = 0; step < interRankSize_ - 1; step++) {
        // 当前轮要接收的分片索引（逆向推导）。
        const u32 recvIdx = (interRank_ + interRankSize_ - step - 1) % interRankSize_;
        const Slice &sendSlice = slices_[sendIdx];
        const Slice &recvSlice = slices_[recvIdx];

        // 1) 通知左邻居：我已准备好（ACK 握手）。
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread_, channelLeft_.handle, NOTIFY_IDX_ACK)));
        // 2) 等待右邻居准备完成。
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread_, channelRight_.handle, NOTIFY_IDX_ACK,
            CUSTOM_TIMEOUT)));

        // 3) 将当前 sendSlice 发送到右邻居。
        // ROCE 路径主动 write 到对端 remoteOutput。
        if (channelRight_.protocol == COMM_PROTOCOL_ROCE) {
            void *dst = static_cast<u8 *>(channelRight_.remoteOutput.addr) + sendSlice.offset + baseOffset_;
            void *src = static_cast<u8 *>(outputMem_.addr) + sendSlice.offset;
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread_, channelRight_.handle, dst, src, sendSlice.size)));
        }
        // 4) 通知右邻居数据已到达（DATA_SIGNAL）。
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread_, channelRight_.handle,
            NOTIFY_IDX_DATA_SIGNAL)));

        // 5) 等待左邻居发送的数据到达。
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread_, channelLeft_.handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        // 非 ROCE 路径在接收侧执行 read（从左邻居 remoteOutput 拉取）。
        if (channelLeft_.protocol != COMM_PROTOCOL_ROCE) {
            void *src = static_cast<u8 *>(channelLeft_.remoteOutput.addr) + recvSlice.offset + baseOffset_;
            void *dst = static_cast<u8 *>(outputMem_.addr) + recvSlice.offset;
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread_, channelLeft_.handle, dst, src, recvSlice.size)));
        }
        // 下一轮继续转发本轮刚接收到的分片。
        sendIdx = recvIdx;
    }

    // 可选 barrier：确保环上全部步骤完成后再返回。
    if (barrierSwitchOn_) {
        CHK_RET(ExecuteBarrier(channelLeft_, channelRight_));
    }
    return HCCL_SUCCESS;
}

// 注册模板实现，供执行器按 TemplateType 动态获取。
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALLGATHER_RING, AllGatherRing);
}
