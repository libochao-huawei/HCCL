#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include <vector>
#include <iostream>
#include <string>

#define private public
#include "testcase_utils.h"
#include "coll_alg_component_builder.h"
#include "virtual_topo.h"
#include "virtual_topo_stub.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "execute_selector.h"
#include "base_selector.h"
#include "dev_buffer.h"
#include "rank_graph_builder.h"
#include "rank_gph.h"
#include "phy_topo_builder.h"
#include "phy_topo.h"
#include "base_selector.h"
#include "dev_capability.h"
#undef private

using namespace Hccl;
using namespace std;

class SelectorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SelectorTest test case set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SelectorTest test case tear down" << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "SelectorTest set up" << std::endl;
        GlobalMockObject::verify();
        InitCollAlgComponents();
    }

    virtual void TearDown()
    {
        std::cout << "SelectorTest tear down" << std::endl;
        SetAlgoEnv("");
        GlobalMockObject::verify();
        ClearHcclEnv();
    }

    void SetAlgoEnv(std::string algo)
    {
        EnvConfig::GetInstance().algoCfg.hcclAlgoConfig.value = SetHcclAlgoConfig(algo);
    }

    CollAlgOperator GetDefaultAlgOp(OpType opType, DataType dataType = DataType::FP16, u64 dataCount = 1024)
    {
        CollAlgOperator collAlgOp;
        collAlgOp.opType = opType;
        collAlgOp.dataType = DataType::FP16;
        collAlgOp.dataCount = dataCount;

        uint64_t dataSize = dataType * DataTypeSizeGet(dataType);
        collAlgOp.inputMem = DevBuffer::Create(0x1000000, dataSize);
        collAlgOp.outputMem = DevBuffer::Create(0x2000000, dataSize);
        collAlgOp.scratchMem = DevBuffer::Create(0x3000000, dataSize);

        if (opType == OpType::ALLREDUCE || opType == OpType::REDUCESCATTER || opType == OpType::REDUCESCATTERV ||
            opType == OpType::REDUCE) {
            collAlgOp.reduceOp = ReduceOp::SUM;
        }
        return collAlgOp;
    }

    std::string GetCommonFilePath()
    {
        std::string localPath = "../test/legacy/st/algorithm/testcase/function_ut_testcase/common_files";
        std::string ciPath = "../test/legacy/st/algorithm/testcase/function_ut_testcase/common_files";

        char buffer[PATH_MAX + 1];  // 使用堆上 buffer, 避免 malloc 和 free 操作
        char *res = realpath(localPath.data(), buffer);
        std::string realRankTableFilePath;
        if (res) {
            return localPath;
        } else {
            return ciPath;
        }
    }

    void InitCollAlgComponent(std::shared_ptr<CollAlgComponent> &component, std::unique_ptr<RankGraph> &rankGraph,
        const std::string &rankTablePath, const std::string &topoPath, uint32_t rankSize)
    {
        if (component != nullptr) {
            return;
        }

        u32 myRank = 0;
        if (commonFilePath_.empty()) {
            commonFilePath_ = GetCommonFilePath();
        }

        rankGraph = BuildRankGraph(commonFilePath_ + rankTablePath, commonFilePath_ + topoPath, myRank);

        CollAlgComponentBuilder builder;
        component = builder.SetRankGraph(rankGraph.get())
                        .SetDevType(DevType::DEV_TYPE_910_95)
                        .SetMyRank(myRank)
                        .SetRankSize(rankSize)
                        .Build();
    }

    void InitCollAlgComponents()
    {
        InitCollAlgComponent(collAlgComponent1d16p_,
            rankGraph1d16p_,
            "/1d_16p_mesh_topo/ranktable.json",
            "/1d_16p_mesh_topo/topo.json",
            16);
        EXPECT_NE(collAlgComponent1d16p_, nullptr);

        InitCollAlgComponent(
            collAlgComponent1d2p_, rankGraph1d2p_, "/1d_2p_mesh_topo/ranktable.json", "/1d_2p_mesh_topo/topo.json", 2);
        EXPECT_NE(collAlgComponent1d2p_, nullptr);

        InitCollAlgComponent(
            collAlgComponent2d2x2p_, rankGraph2d2x2p_, "/2d_2+2p_mesh_topo/ranktable.json", "/2d_2+2p_mesh_topo/topo.json", 4);
        EXPECT_NE(collAlgComponent2d2x2p_, nullptr);

        InitCollAlgComponent(
            collAlgComponent2pClos_, rankGraph2pClos_, "/2p_clos_topo/ranktable.json", "/2p_clos_topo/topo.json", 2);
        EXPECT_NE(collAlgComponent2pClos_, nullptr);

        InitCollAlgComponent(
            collAlgComponent1d2pMeshClos_, rankGraph1d2pMeshClos_, "/1d_2p_mesh_clos_topo/ranktable.json", "/1d_2p_mesh_clos_topo/topo.json", 2);
        EXPECT_NE(collAlgComponent1d2pMeshClos_, nullptr);
    }

    // CI 上的文件位置和本地执行时有所不同，本地执行为：../llt/xxx, CI 为：./llt/xxx
    std::string commonFilePath_ = "";
    unique_ptr<RankGraph> rankGraph1d16p_ = nullptr;
    unique_ptr<RankGraph> rankGraph1d2p_ = nullptr;
    unique_ptr<RankGraph> rankGraph2d2x2p_ = nullptr;
    unique_ptr<RankGraph> rankGraph2pClos_ = nullptr;
    unique_ptr<RankGraph> rankGraph1d2pMeshClos_ = nullptr;
    std::shared_ptr<CollAlgComponent> collAlgComponent1d16p_ = nullptr;
    std::shared_ptr<CollAlgComponent> collAlgComponent1d2p_ = nullptr;
    std::shared_ptr<CollAlgComponent> collAlgComponent2d2x2p_ = nullptr;
    std::shared_ptr<CollAlgComponent> collAlgComponent2pClos_ = nullptr;
    std::shared_ptr<CollAlgComponent> collAlgComponent1d2pMeshClos_ = nullptr;
};

TEST_F(SelectorTest, SelectorTest)
{
    // ========================
    // 可配置参数区域
    // ========================
    std::vector<OpType> opTypes = {
        OpType::ALLREDUCE,
        OpType::ALLGATHER,
        OpType::REDUCESCATTER,
        OpType::BROADCAST,
        OpType::ALLTOALL,
        OpType::ALLTOALLV,
        OpType::ALLGATHERV,
        OpType::REDUCESCATTERV,
        OpType::REDUCE,
        OpType::SCATTER,
        OpType::BATCHSENDRECV,
        OpType::ALLTOALLVC
    };

    std::vector<AcceleratorState> accStates = {
        AcceleratorState::CCU_MS,
        AcceleratorState::CCU_SCHED,
        AcceleratorState::AIV,
        AcceleratorState::CCU_FALLBACK,
        AcceleratorState::AICPU_TS
    };

    // 保持原来的 topoComponents 结构不变，避免对不上的问题
    std::vector<std::pair<std::string, std::shared_ptr<CollAlgComponent>>> topoComponents = {
        {"1d16p", collAlgComponent1d16p_},
        {"1d2p", collAlgComponent1d2p_},
        {"2d2x2p", collAlgComponent2d2x2p_},
        {"2pClos", collAlgComponent2pClos_},
        {"1d2pMeshClos", collAlgComponent1d2pMeshClos_},
    };

    std::vector<DataType> dataTypes = {
        // DataType::INT8,
        DataType::FP16,
        // DataType::INT64
    };

    std::vector<u64> dataCounts = {
        1024,
        1024 * 1024,
        1024 * 1024 * 1024
    };

    // ========================
    // 测试逻辑
    // ========================
    CollAlgParams collAlgParams;
    collAlgParams.opMode = OpMode::OPBASE;

    auto runTest = [&](const std::string& topoName, const std::shared_ptr<CollAlgComponent>& component) {
        for (auto opType : opTypes) {
            for (auto accState : accStates) {
                collAlgParams.opExecuteConfig.accState = accState;
                for (auto dataType : dataTypes) {
                    for (auto dataCount : dataCounts) {
                        std::cout << "Testing: topo=" << topoName
                                  << ", accState=" << accState.Describe()
                                  << ", opType=" << opType.Describe()
                                  << ", dataType=" << dataType.Describe()
                                  << ", dataCount=" << dataCount << std::endl;

                        CollAlgOperator collAlgOp = GetDefaultAlgOp(opType, dataType, dataCount);
                        std::string algName = "";
                        EXPECT_NO_THROW(component->ExecAlgSelect(
                            collAlgOp, collAlgParams, algName, collAlgParams.opExecuteConfig));

                        if (algName.empty()) {
                            algName = "FAILED";
                        }
                        std::cout << "CCH res: ," << topoName
                                  << ", " << opType.Describe()
                                  << ", " << accState.Describe()
                                  << ", " << dataType.Describe()
                                  << ", " << dataCount
                                  << ", " << algName << std::endl;
                    }
                }
            }
        }
    };

    for (const auto& topo : topoComponents) {
        runTest(topo.first, topo.second);
    }
}

TEST_F(SelectorTest, SelectorMc2Test)
{
    OpExecuteConfig opExecuteConfig;
    opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    CollAlgParams collAlgParams;
    collAlgParams.opMode = OpMode::OFFLOAD;
    collAlgParams.opExecuteConfig = opExecuteConfig;
    collAlgParams.isMc2 = true;
    std::string algName = "";

    CollAlgOperator collAlgOp = GetDefaultAlgOp(OpType::ALLGATHER);

    EXPECT_NO_THROW(collAlgComponent1d2p_->ExecAlgSelect(collAlgOp, collAlgParams, algName, opExecuteConfig));

    opExecuteConfig.accState = AcceleratorState::CCU_MS;
    collAlgParams.opExecuteConfig = opExecuteConfig;
    EXPECT_NO_THROW(collAlgComponent1d2p_->ExecAlgSelect(collAlgOp, collAlgParams, algName, opExecuteConfig));
}

TEST_F(SelectorTest, SelectorMc2Test2D)
{
    OpExecuteConfig opExecuteConfig;
    opExecuteConfig.accState = AcceleratorState::AICPU_TS;

    CollAlgParams collAlgParams;
    collAlgParams.opMode = OpMode::OFFLOAD;
    collAlgParams.opExecuteConfig = opExecuteConfig;
    collAlgParams.isMc2 = true;
    std::string algName = "";

    CollAlgOperator collAlgOp = GetDefaultAlgOp(OpType::ALLGATHER);

    EXPECT_NO_THROW(collAlgComponent2d2x2p_->ExecAlgSelect(collAlgOp, collAlgParams, algName, opExecuteConfig));

    opExecuteConfig.accState = AcceleratorState::CCU_MS;
    collAlgParams.opExecuteConfig = opExecuteConfig;
    EXPECT_NO_THROW(collAlgComponent2d2x2p_->ExecAlgSelect(collAlgOp, collAlgParams, algName, opExecuteConfig));
}

TEST_F(SelectorTest, TestAutoSelectorCcuSchedule_1D_2P)
{
    RankGraph *virtTopo = rankGraph1d2p_.get();
    u32 rankSize = 2;
    ExecuteSelector selector = ExecuteSelector().SetVirtualTopo(virtTopo).SetMyRank(0).SetRankSize(rankSize);

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::CCU_SCHED;

    CollAlgParams params;
    params.opExecuteConfig = opConfig;

    std::string selectedAlgName;
    CollAlgOperator collAlgOp;

    collAlgOp = GetDefaultAlgOp(OpType::ALLREDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllReduceMeshMem2Mem1D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllGatherMeshMem2Mem1D");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceScatterMeshMem2Mem1D");

    collAlgOp = GetDefaultAlgOp(OpType::BROADCAST);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuBroadcastMeshMem2Mem1D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALL);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAlltoAllMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAlltoAllVMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllGatherVMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceScatterVMeshMem2Mem1D");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceMeshMem2Mem1D");

    collAlgOp = GetDefaultAlgOp(OpType::SCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuScatterMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::BATCHSENDRECV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsBatchSendRecv");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLVC);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAlltoAllvcMesh");
}

TEST_F(SelectorTest, TestAutoSelectorCcuSchedule_2D_2x2P)
{
    RankGraph *virtTopo = rankGraph2d2x2p_.get();
    u32 rankSize = 4;
    ExecuteSelector selector = ExecuteSelector().SetVirtualTopo(virtTopo).SetMyRank(0).SetRankSize(rankSize);

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::CCU_SCHED;

    CollAlgParams params;
    params.opExecuteConfig = opConfig;

    std::string selectedAlgName;
    CollAlgOperator collAlgOp;

    collAlgOp = GetDefaultAlgOp(OpType::ALLREDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllReduceMeshTwoShotMem2Mem2D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllGatherMeshMem2Mem2D");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceScatterMeshMem2Mem2D");

    collAlgOp = GetDefaultAlgOp(OpType::BROADCAST);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuBroadcastMeshMem2Mem2D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALL);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAlltoAllMesh2D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAlltoAllVMesh2D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_NE(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_NE(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);

    collAlgOp = GetDefaultAlgOp(OpType::REDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceMeshMem2Mem2D");

    collAlgOp = GetDefaultAlgOp(OpType::SCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuScatterMesh2D");

    collAlgOp = GetDefaultAlgOp(OpType::BATCHSENDRECV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsBatchSendRecv");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLVC);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAlltoAllvcMesh");
}

TEST_F(SelectorTest, TestAutoSelectorCcuMS_1D_2P)
{
    RankGraph *virtTopo = rankGraph1d2p_.get();
    u32 rankSize = 2;
    ExecuteSelector selector = ExecuteSelector().SetVirtualTopo(virtTopo).SetMyRank(0).SetRankSize(rankSize);

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::CCU_MS;

    CollAlgParams params;
    params.opExecuteConfig = opConfig;

    std::string selectedAlgName;
    CollAlgOperator collAlgOp;

    collAlgOp = GetDefaultAlgOp(OpType::ALLREDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllReduceMesh1DOneShot");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllGatherMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceScatterMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::BROADCAST);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuBroadcastMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALL);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAlltoAllMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAlltoAllVMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuAllGatherVMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceScatterVMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuReduceMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::SCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "CcuScatterMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::BATCHSENDRECV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsBatchSendRecv");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLVC);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAlltoAllvcMesh");
}

TEST_F(SelectorTest, TestAutoSelectorCcuMS_Clos)
{
    RankGraph *virtTopo = rankGraph2pClos_.get();
    u32 rankSize = 2;
    ExecuteSelector selector = ExecuteSelector().SetVirtualTopo(virtTopo).SetMyRank(0).SetRankSize(rankSize);

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::CCU_MS;

    CollAlgParams params;
    params.opExecuteConfig = opConfig;

    std::string selectedAlgName;
    CollAlgOperator collAlgOp;

    collAlgOp = GetDefaultAlgOp(OpType::ALLREDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAllReduceNHR");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAllGatherNHR");

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsReduceScatterNHR");

    collAlgOp = GetDefaultAlgOp(OpType::BROADCAST);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsBroadcastNHR");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALL);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAlltoAllMesh");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAlltoAllvMesh");

    collAlgOp = GetDefaultAlgOp(OpType::ALLGATHERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_NE(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);

    collAlgOp = GetDefaultAlgOp(OpType::REDUCESCATTERV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_NE(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);

    collAlgOp = GetDefaultAlgOp(OpType::REDUCE);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsReduceNHR");

    collAlgOp = GetDefaultAlgOp(OpType::SCATTER);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsScatterMesh1D");

    collAlgOp = GetDefaultAlgOp(OpType::BATCHSENDRECV);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsBatchSendRecv");

    collAlgOp = GetDefaultAlgOp(OpType::ALLTOALLVC);
    selectedAlgName = "";
    params.opExecuteConfig = opConfig;
    EXPECT_EQ(selector.Run(collAlgOp, params, selectedAlgName), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(selectedAlgName, "InsAlltoAllvcMesh");
}
