#include "hccl_ipc.h"
#include "hccl_ipc_comm.h"

using namespace HcclSim;

HcclIpcComm g_hcclComm;

HcclVmResult HcclIpcPushRequest(const HcclTaskReq &req)
{
    return g_hcclComm.Push(req);
}

HcclVmResult HcclIpcPullResponse(uint16_t rankId, HcclTaskRsp &rsp)
{
    return g_hcclComm.Pull(rankId, rsp);
}

HcclVmResult HcclIpcPushResponse(const HcclTaskRsp &rsp)
{
    return g_hcclComm.Push(rsp);
}

HcclVmResult HcclIpcPullRequest(HcclTaskReq &req)
{
    return g_hcclComm.Pull(req);
}