#include <stdio.h>
#include "ut_hccl_ipc.h"
#include <boost/interprocess/ipc/message_queue.hpp>
#include "hccl_common_defs.h"
#include "hccl_ipc.h"

#define private public

#include "hccl_ipc_comm.h"

using namespace boost::interprocess;
using namespace HcclSim;

extern HcclIpcComm g_hcclComm;

void InitIpc()
{
    message_queue::remove("runtime_0");
    message_queue mqRsp(create_only, "runtime_0", 10240, sizeof(HcclTaskReq));
    message_queue::remove("proxy_0");
    message_queue mqReq(create_only, "proxy_0", 10240, sizeof(HcclTaskRsp));
}

TEST_F(HcclIpcSuite, HcclIpcPushRequest)
{
    InitIpc();
    HcclTaskReq req = {0, 0};
    g_hcclComm.m_mode = 0;
    HcclVmResult ret = HcclIpcPushRequest(req);
    g_hcclComm.m_mode = 1;
    ret = HcclIpcPullRequest(req);
    EXPECT_EQ(ret, 0);
}


