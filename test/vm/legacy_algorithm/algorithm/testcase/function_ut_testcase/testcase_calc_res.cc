#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include <vector>
#include <iostream>
#include <string>

#include "testcase_utils.h"
#include "coll_alg_component_builder.h"
#include "virtual_topo.h"
#include "virtual_topo_stub.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "execute_selector.h"
#include "base_selector.h"
#include "dev_buffer.h"


using namespace Hccl;
using namespace std;

class CalcResTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CalcResTest set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CalcResTest tear down" << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "CalcResTest set up" << std::endl;
    }

    virtual void TearDown()
    {
        ClearHcclEnv();
    }
};

TEST_F(CalcResTest, CalcResOffloadTest)
{
    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095OneTimesFour(rankTable);

    RankId myRank = 0;
    u32 rankSize = 4;

    CollAlgComponentBuilder collAlgComponentBuilder;
    std::shared_ptr<CollAlgComponent> collAlgComponent = collAlgComponentBuilder.SetRankGraph(&virtTopo)
                                                             .SetDevType(DevType::DEV_TYPE_910_95)
                                                             .SetMyRank(myRank)
                                                             .SetRankSize(rankSize)
                                                             .Build();
    OpExecuteConfig opExecuteConfig;
    AcceleratorState accState = AcceleratorState::CCU_MS;
    setenv("PRIM_QUEUE_GEN_NAME", "CcuAllReduceMesh1D", 1);

    OpType opType = OpType::ALLREDUCE;
    u64 dataSize = 100;
    CollOffloadOpResReq resReq1;

    EXPECT_NO_THROW(collAlgComponent->CalcResOffload(opType, dataSize, HcclDataType::HCCL_DATA_TYPE_INT16, opExecuteConfig, resReq1));
    EXPECT_EQ(16, resReq1.requiredSubQueNum);
    EXPECT_EQ(256 * 1024 * 1024, resReq1.requiredScratchMemSize);
    unsetenv("PRIM_QUEUE_GEN_NAME");
}


