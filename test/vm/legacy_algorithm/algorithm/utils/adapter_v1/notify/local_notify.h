/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: local notify interface
 */

#ifndef TEST_LOCAL_NOTIFY_H
#define TEST_LOCAL_NOTIFY_H

#include "stream_pub.h"
#include "dispatcher.h"
#include "hccl_common.h"

namespace hccl {
class LocalNotify {
public:
    LocalNotify() = default;
    ~LocalNotify() = default;

    HcclResult Init(const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY);

    static HcclResult Wait(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify,
        s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME);
    static HcclResult Post(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify,
        s32 stage = INVALID_VALUE_STAGE);

    HcclResult Wait(Stream& stream, HcclDispatcher dispatcherPtr,
        s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME);
    HcclResult Post(Stream& stream, HcclDispatcher dispatcherPtr,
        s32 stage = INVALID_VALUE_STAGE);

    u32 GetNotifyIdx();
    HcclResult SetNotifyId(u32 notifyId);

    HcclResult Destroy();
    HcclResult SetIpc();
    HcclResult GetNotifyData(HcclSignalInfo &notifyInfo);

    inline HcclRtNotify ptr()
    {
        return notifyPtr;
    }

    HcclRtNotify notifyPtr = nullptr;
private:
    u32 notifyidx_;
};
}

#endif // LOCAL_NOTIFY_H
