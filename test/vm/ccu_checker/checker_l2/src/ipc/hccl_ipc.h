#ifndef HCCL_IPC_H
#define HCCL_IPC_H

#include <string>
#include "hccl_common_defs.h"

// 将任务请求推送到IPC通道
HcclSim::HcclVmResult HcclIpcPushRequest(const HcclTaskReq& req);
// 从IPC通道拉取任务响应
HcclSim::HcclVmResult HcclIpcPullResponse(uint16_t rankId, HcclTaskRsp &rsp);

// 将任务响应推送到IPC通道
HcclSim::HcclVmResult HcclIpcPushResponse(const HcclTaskRsp& rsp);
// 从IPC通道拉取任务请求
HcclSim::HcclVmResult HcclIpcPullRequest(HcclTaskReq& req);

#endif