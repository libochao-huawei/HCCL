/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2024. All rights reserved.
 * Description: stream管理操作类
 * Author: mali
 * Create: 2018-02-20
 */

#include "stream.h"

namespace hccl {
Stream::Stream()
{
}


Stream::~Stream()
{
}

Stream::Stream(const Stream &that)
    : stream_(that.ptr()), isMainStream_(that.isMainStream_), streamId_(that.streamId_) {
    }

Stream::Stream(Stream &&that)
    : stream_(that.ptr()), isMainStream_(that.isMainStream_), streamId_(that.streamId_)
{
    that.stream_ = nullptr;
}

Stream::Stream(const rtStream_t rtStream, bool isMainStream)
    : stream_(const_cast<void *>(rtStream)), isMainStream_(isMainStream)
{
}

Stream::Stream(const StreamType streamType, bool isMainStream)
{}

Stream Stream::operator=(Stream &&that)
{
    if (&that != this) {
        stream_ = that.stream_;
        isMainStream_ = that.isMainStream_;
        streamId_ = that.streamId_;
    }
    that.stream_ = nullptr;
    return *this;
}

Stream &Stream::operator=(const Stream &that)
{
    if (&that != this) {
        stream_ = that.ptr();
        isMainStream_ = that.isMainStream_;
        streamId_ = that.streamId_;
    }
    return *this;
}

HcclResult Stream::PopTaskLogicInfo(TaskLogicInfo &taskLogicInfo)
{
    return HCCL_SUCCESS;
}

StreamAddrRecorder* StreamAddrRecorder::Global()
{
    static StreamAddrRecorder* streamAddrRecorder = new StreamAddrRecorder;
    return streamAddrRecorder;
}

}  // namespace hccl
