#include "hccl_ipc.h"

using namespace HcclSim;

HcclVmResult HcclIpcPushRequest(const HcclTaskReq &req)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclIpcPullResponse(HcclTaskRsp &rsp)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclIpcPushResponse(const HcclTaskRsp &rsp)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclIpcPullRequest(HcclTaskReq &req)
{
    return HcclVmResult::HCCL_SIM_SUCCESS;
}