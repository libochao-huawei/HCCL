#ifndef HCCL_ALLGATHERBATCH_RESOURCE_REQUEST_H
#define HCCL_ALLGATHERBATCH_RESOURCE_REQUEST_H

#include <vector>

#include "common.h"

namespace ops_hccl_allgatherbatch {

struct ChannelRequest {
    uint32_t remoteRank = 0;
    uint32_t remoteServerIdx = 0;
    uint32_t remoteSuperPodIdx = 0;
    CommProtocol protocol = COMM_PROTOCOL_RESERVED;
    uint32_t notifyNum = 0;
};

// Host 侧资源请求。`GetAlgRes` 只负责按请求分配资源，不再自己隐式推导 channel 模型。
// 当前前提固定为 fullmesh，因此 channel 数量始终等于 rankSize - 1。
struct BatchResourceRequest {
    uint32_t threadNum = 1;
    uint32_t controlNotifyNum = kAllGatherBatchControlNotifyNum;
    uint32_t mainThreadNotifyNum = 0;
    uint32_t lastTwoWorkerCount = 0;
    uint32_t workerNotifyNum = 0;
    uint32_t channelCount = 0;
    uint64_t localBufferBytes = 0;
    BatchCommMode commMode = BatchCommMode::kUnknown;
    std::vector<ChannelRequest> channels;
};

}  // namespace ops_hccl_allgatherbatch

#endif
