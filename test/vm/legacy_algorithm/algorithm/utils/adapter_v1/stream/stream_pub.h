/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2022. All rights reserved.
 * Description: stream管理操作类对外头文件
 */

#ifndef STREAM_PUB_H
#define STREAM_PUB_H

#include <memory>
#include <mutex>
#include <queue>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "task_logic_info_pub.h"

namespace hccl {
constexpr int32_t HCCL_STREAM_PRIORITY_LOW = 0;
constexpr int32_t HCCL_STREAM_PRIORITY_HIGH = 0;
enum class StreamType {
    STREAM_TYPE_OFFLINE = 0,
    STREAM_TYPE_ONLINE = 1,
    STREAM_TYPE_DEVICE = 2,
    STREAM_TYPE_RESERVED = 3
};

/*
 * NOTE : hccl中, 节点内device间的link都有自己的event. 当前约定:
 * link对象作为发送方时record自己的event
 * link对象作为接收方时wait发送方的event
 */
class Stream {
public:

    Stream();
    Stream(const Stream &that);
    Stream(Stream &&that);
    explicit Stream(const StreamType streamType, bool isMainStream = false);
    explicit Stream(const rtStream_t rtStream, bool isMainStream = true);

    ~Stream();

    bool isMainStream_ = false;
    inline bool IsMainStream()
    {
        return isMainStream_;
    }

    Stream &operator=(const Stream &that);
    Stream operator=(Stream &&that);

    void *ptr() const
    {
        return stream_;
    }
    operator bool() const
    {
        return stream_ != nullptr;
    }
    s32 id() const
    {
        return streamId_;
    }
    u32 sqId() const
    {
        return sqId_;
    }

    void *stream_ = nullptr;
    s32 streamId_ = -1;
    u32 sqId_;

    HcclResult PopTaskLogicInfo(TaskLogicInfo &taskLogicInfo);
};

class StreamAddrRecorder {
public:
    static StreamAddrRecorder* Global();
    u64 streamAddr = 0x1;
};

}  // namespace hccl

#endif /* STREAM_PUB_H */