
#include "communicator_impl_stub.h"

#include <cstdint>

#include "log.h"
#include "coll_service_stub.h"
#include "env_config.h"
#include "env_func.h"
#include "dev_capability.h"
#include "rank_graph_builder.h"
#include "coll_gen_json.h"
#include "mem_layout.h"
#include "llt_common.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator.h"
#include "ccu_component.h"
#include "rdma_handle_manager.h"

using namespace std;
namespace Hccl {

void PrintBackTrace(HcclException &e)
{
    auto backTraces = e.GetBackTraceStrings();
    std::for_each(backTraces.begin(), backTraces.end(), [](string item) {
        HCCL_DEBUG(item.c_str());
    });
}

constexpr u64 HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE = (1 * 1024 * 1024); // 指定bufferSize的单位为MB


u32 GetLocalDieId(IpAddress&& portAddr)
{
    auto     devLogicId = HrtGetDevice();
    uint32_t devPhyId   = HrtGetDevicePhyIdByIndex(devLogicId);
 
    auto &rdmaHandleMgr = RdmaHandleManager::GetInstance();
    auto  rdmaHandle    = rdmaHandleMgr.GetByIp(devPhyId, portAddr);
    auto  dieId         = rdmaHandleMgr.GetDieAndFuncId(rdmaHandle).first;
    return dieId;
}

HcclResult CommunicatorImpl::Init(const CommParams &commParams,
    const std::string &ranktableM, std::string& topoPath,
    const HcclCommConfig &config)
{
    if (!initFlag) {
        initFlag = true;
        try {
            // 判断开启哪些特性
            InitFeatureFlag();
            InitCommonData(commParams, config);
            InitCcuInstance(); // 依赖获取devLogicId ，需要在初始化InitCommonData之后调用
            InitRankGraph(ranktableM, topoPath);
            AppendLocalDieIdForLinks();
            InitCollService();
            status = CommStatus::COMM_READY;
        } catch (HcclException &e) {
            HCCL_ERROR(e.what());
            PrintBackTrace(e);
            return e.GetErrorCode();
        } catch (exception &e) {
            HCCL_ERROR(e.what());
            return HcclResult::HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("Unknown error occurs!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_ERROR("Repeated calling init method!");
    return HcclResult::HCCL_E_INTERNAL;
}

CollOperator *CommunicatorImpl::GetCurrentCollOperator() const
{
    CHECK_NULLPTR(currentCollOperator, "currentCollOperator is nullptr!");
    return currentCollOperator.get();
}

HcclResult CommunicatorImpl::LoadOpbasedCollOp(CollOpParams &opParams, std::string& algName)
{
    try {
        if (status == CommStatus::COMM_ERROR) {
            HCCL_ERROR("Comm has been error, can not load opbased operator now!");
            return HcclResult::HCCL_E_INTERNAL;
        }

        CovertToCurrentCollOperator(id, opParams, OpMode::OPBASE);
        collService->LoadWithOpBasedMode(*currentCollOperator, algName);
    } catch (HcclException &e) {
        status = CommStatus::COMM_ERROR;
        HCCL_ERROR(e.what());
        PrintBackTrace(e);
        HCCL_ERROR("OperatorParams: %s", opParams.Describe().c_str());
        return e.GetErrorCode();
    } catch (exception &e) {
        status = CommStatus::COMM_ERROR;
        HCCL_ERROR(e.what());
        HCCL_ERROR("OperatorParams: %s", opParams.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        status = CommStatus::COMM_ERROR;
        HCCL_ERROR("OperatorParams: %s", opParams.Describe().c_str());
        HCCL_ERROR("Unkown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::LoadOffloadCollOp(std::string &opTag, CollOpParams &opParams)
{
    try {
        if (status == CommStatus::COMM_ERROR) {
            HCCL_ERROR("Comm has been error, can not offload operator now!");
            return HcclResult::HCCL_E_INTERNAL;
        }

        CovertToCurrentCollOperator(opTag, opParams, OpMode::OFFLOAD);
        collService->LoadWithOffloadMode(*currentCollOperator);
    } catch (HcclException &e) {
        status = CommStatus::COMM_ERROR;
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception &e) {
        status = CommStatus::COMM_ERROR;
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        status = CommStatus::COMM_ERROR;
        HCCL_ERROR("Unkown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

constexpr u32 CCL_COMM_DEFAULT_BUFFER_SIZE    = 200;
constexpr u64 CCL_COMM_FIXED_CALC_BUFFER_SIZE = (1 * 1024 * 1024);
void CommunicatorImpl::CalcA2ASendRecvMem(CollOpParams &opParams, u64 &sendSize, u64 &recvSize) const
{
    u64 sendCount = 0;
    u64 recvCount = 0;
    u32 sendTypeSize = 0;
    u32 recvTypeSize = 0;
    if (opParams.opType == OpType::ALLTOALLV) {
        for (u32 i = 0; i < rankSize; i++) {
            u64 curSendCount = *(static_cast<const u64 *>(opParams.all2AllVDataDes.sendCounts) + i) +
                *(static_cast<const u64 *>(opParams.all2AllVDataDes.sdispls) + i);
            sendCount = std::max(sendCount, curSendCount);
            u64 curRecvCount = *(static_cast<const u64 *>(opParams.all2AllVDataDes.recvCounts) + i) +
                *(static_cast<const u64 *>(opParams.all2AllVDataDes.rdispls) + i);
            recvCount = std::max(recvCount, curRecvCount);
        }
        sendTypeSize = DataTypeSizeGet(opParams.all2AllVDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(opParams.all2AllVDataDes.recvType);
    } else if (opParams.opType == OpType::ALLTOALLVC){
        for (u32 i = 0; i < rankSize; i++) {
            sendCount += *(static_cast<const u64 *>(opParams.all2AllVCDataDes.sendCountMatrix) +
                            myRank * rankSize + i);
            recvCount += *(static_cast<const u64 *>(opParams.all2AllVCDataDes.sendCountMatrix) +
                            myRank + rankSize * i);
        }
        sendTypeSize = DataTypeSizeGet(opParams.all2AllVCDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(opParams.all2AllVCDataDes.recvType);
    } else {
        sendCount = opParams.all2AllDataDes.sendCount * rankSize;
        recvCount = opParams.all2AllDataDes.recvCount * rankSize;
        sendTypeSize = DataTypeSizeGet(opParams.all2AllDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(opParams.all2AllDataDes.recvType);
    }
    sendSize = sendCount * sendTypeSize;
    recvSize = recvCount * recvTypeSize;
}

void CommunicatorImpl::ConvertCollOperatorA2A(CollOpParams &opParams)
{
    if (currentCollOperator) {
        HCCL_DEBUG("ConvertCollOperatorA2A START");
        if (opParams.opType == OpType::ALLTOALL) {
            currentCollOperator->all2AllDataDes.sendCount = opParams.all2AllDataDes.sendCount;
            currentCollOperator->all2AllDataDes.recvCount = opParams.all2AllDataDes.recvCount;
            currentCollOperator->all2AllDataDes.sendType = opParams.all2AllDataDes.sendType;
            currentCollOperator->all2AllDataDes.recvType = opParams.all2AllDataDes.recvType;
            HCCL_DEBUG("sendCount[%llu], recvCount[%llu]", opParams.all2AllDataDes.sendCount, opParams.all2AllDataDes.recvCount);
        } else if (opParams.opType == OpType::ALLTOALLV) {
            currentCollOperator->all2AllVDataDes.sendCounts = opParams.all2AllVDataDes.sendCounts;
            currentCollOperator->all2AllVDataDes.recvCounts = opParams.all2AllVDataDes.recvCounts;
            currentCollOperator->all2AllVDataDes.sdispls = opParams.all2AllVDataDes.sdispls;
            currentCollOperator->all2AllVDataDes.rdispls = opParams.all2AllVDataDes.rdispls;
            currentCollOperator->all2AllVDataDes.sendType = opParams.all2AllVDataDes.sendType;
            currentCollOperator->all2AllVDataDes.recvType = opParams.all2AllVDataDes.recvType;
        } else if (opParams.opType == OpType::ALLTOALLVC) {
            currentCollOperator->all2AllVCDataDes.sendType = opParams.all2AllVCDataDes.sendType;
            currentCollOperator->all2AllVCDataDes.recvType = opParams.all2AllVCDataDes.recvType;
            currentCollOperator->all2AllVCDataDes.sendCountMatrix = opParams.all2AllVCDataDes.sendCountMatrix;
        }
        u64 sendSize = 0;
        u64 recvSize = 0;
        CalcA2ASendRecvMem(opParams, sendSize, recvSize);
        HCCL_DEBUG("sendSize[%llu], recvSize[%llu]", sendSize, recvSize);
        currentCollOperator->inputMem  = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), sendSize);
        currentCollOperator->outputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), recvSize);
    } else {
        HCCL_ERROR("currentCollOperator is nullptr");
    }
}

void CommunicatorImpl::ConvertCollOperatorMem(CollOpParams &opParams, u64 size)
{
    HCCL_DEBUG("[CommunicatorImpl::%s] start, opType[%s], size[%llu]", __func__, opParams.opType.Describe().c_str(), size);

    if (opParams.opType == OpType::REDUCESCATTER || opParams.opType == OpType::SCATTER) {
        currentCollOperator->inputMem
            = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), size * rankSize);
    } else {
        currentCollOperator->inputMem
            = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), size);
    }

    if (opParams.opType == OpType::ALLGATHER || opParams.opType == OpType::GATHER) {
        currentCollOperator->outputMem
            = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), size * rankSize);
    } else {
        currentCollOperator->outputMem
            = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), size);
    }

    HCCL_DEBUG("[CommunicatorImpl::%s] end.", __func__);
}

void CommunicatorImpl::CovertToCurrentCollOperator(std::string &opTag, CollOpParams &opParams, OpMode opMode)
{
    HCCL_INFO("[CommunicatorImpl::%s] start", __func__);
    currentCollOperator = make_unique<CollOperator>();
    if (!currentCollOperator) {
        HCCL_ERROR("[CommunicatorImpl::%s] currentCollOperator is nullptr", __func__);
    }
    currentCollOperator->opMode      = opMode;
    currentCollOperator->opTag       = opTag; // 单算子 标签 为通信域id, 图模式 标签 为传入的opTag
    currentCollOperator->staticAddr  = opParams.staticAddr;
    currentCollOperator->staticShape = opParams.staticShape;
    HCCL_INFO("[CommunicatorImpl::%s] scratchMem start :opMode[%s]", __func__, currentCollOperator->opMode.Describe().c_str());
    if (opMode == OpMode::OPBASE) { // 单算子Scratch buffer为CCL Buffer
        MemBlock block = MemLayout::Global()->GetMemBlock(checker::BufferType::SCRATCH, myRank);
        currentCollOperator->scratchMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(block.startAddr), block.size);
    } else if (opMode == OpMode::OFFLOAD) {
        // TODO: 图模式的scratch大小如何确定呢
        MemBlock block = MemLayout::Global()->GetMemBlock(checker::BufferType::SCRATCH, myRank);
        currentCollOperator->scratchMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(block.startAddr), block.size);
    }

    currentCollOperator->opType    = opParams.opType;
    currentCollOperator->reduceOp  = opParams.reduceOp;
    currentCollOperator->root      = opParams.root;
    currentCollOperator->outputDataType = opParams.outputDataType;
    currentCollOperator->debugCase = opParams.debugCase;
    currentCollOperator->sendRecvRemoteRank = opParams.dstRank;
    if (opParams.opType == OpType::ALLTOALL || opParams.opType == OpType::ALLTOALLV || opParams.opType == OpType::ALLTOALLVC) {
        ConvertCollOperatorA2A(opParams);
    } else if (opParams.opType == OpType::BATCHSENDRECV) {
        currentCollOperator->batchSendRecvDataDes.sendRecvItemsPtr = opParams.batchSendRecvDataDes.sendRecvItemsPtr;
        currentCollOperator->batchSendRecvDataDes.itemNum = opParams.batchSendRecvDataDes.itemNum;
        HCCL_DEBUG("[CommunicatorImpl::%s] OpType::BATCHSENDRECV item = %llu", __func__, currentCollOperator->batchSendRecvDataDes.itemNum);
    } else if (opParams.opType == OpType::REDUCESCATTERV || opParams.opType == OpType::ALLGATHERV) {
        currentCollOperator->vDataDes.counts = opParams.vDataDes.counts;
        currentCollOperator->vDataDes.displs = opParams.vDataDes.displs;
        currentCollOperator->vDataDes.dataType = opParams.vDataDes.dataType;
        currentCollOperator->dataType = opParams.dataType;

        u64 myRankBuffSize = static_cast<u64 *>(opParams.vDataDes.counts)[myRank];
        u64 allRankBuffSize = 0;
        u64 maxDispls = 0;
        for (u32 i = 0; i < rankSize; i++) {
            if (maxDispls < static_cast<u64 *>(opParams.vDataDes.displs)[i]) {
                maxDispls = static_cast<u64 *>(opParams.vDataDes.displs)[i];
                allRankBuffSize = maxDispls + static_cast<u64 *>(opParams.vDataDes.counts)[i];
            }
        }
        u64 sendBuffSize;
        u64 recvBuffSize;
        if (opParams.opType == OpType::ALLGATHERV) {
            sendBuffSize = myRankBuffSize;
            recvBuffSize = allRankBuffSize;
        } else if (opParams.opType == OpType::REDUCESCATTERV) {
            sendBuffSize = allRankBuffSize;
            recvBuffSize = myRankBuffSize;
        }
        currentCollOperator->inputMem  = DevBuffer::Create(reinterpret_cast<uintptr_t >(opParams.sendBuf), sendBuffSize);
        currentCollOperator->outputMem = DevBuffer::Create(reinterpret_cast<uintptr_t >(opParams.recvBuf), recvBuffSize);
    } else {
        currentCollOperator->dataType  = opParams.dataType;
        currentCollOperator->dataCount = opParams.count;

        u64 size = DataTypeSizeGet(opParams.dataType) * opParams.count;
        if (size != 0) {
            ConvertCollOperatorMem(opParams, size);
        } else {
            HCCL_WARNING("[CommunicatorImpl::%s] size is 0", __func__);
        }
    }
}

void CommunicatorImpl::InitCommonData(const CommParams &commParams, const HcclCommConfig &commConfig)
{
    // auto hcclOpMode = EnvConfig::GetInstance().GetAlgoConfig().GetOpExpansionMode();
    id      = commParams.commId;
    myRank                 = commParams.myRank;
    rankSize               = commParams.rankSize;
    devType                = commParams.devType;
    isWorldGroup           = commParams.isWorldGroup;
    devLogicId             = HrtGetDevice();
    devPhyId               = HrtGetDevicePhyIdByIndex(devLogicId);
    config                 = commConfig;
    cclBufferSize          = config.hcclBufferSize;
    // 设定devType，初始化能力，算法及其他模块通过Get获取能力
    // 这样怎么支持混合组网呢？
    DevCapability::GetInstance().Init(devType);
}

void CommunicatorImpl::CheckVirtualTopo() const
{
    // 校验虚拟拓扑中的rankSize和通信域的rankSize一致
    u32 virtRankSize = newVirtualTopo->GetRankSize();
    if (virtRankSize != rankSize) {
        std::string msg
            = StringFormat("Check rankGraph failed, communicator rankSize[%u] does not equal rankTable rankSize[%u]",
                           rankSize, virtRankSize);
        THROW<InvalidParamsException>(msg);
    }

}

void CommunicatorImpl::InitRankGraph(const string &ranktableM, std::string& topoPath)
{
    RankGraphBuilder virtTopoBuilder;

    checker::HcclUs startut = TIME_NOW();
    newVirtualTopo = virtTopoBuilder.Build(ranktableM, topoPath, myRank);
    HCCL_INFO("virtTopoBuilder take time [%lld]us.", DURATION_US(TIME_NOW() - startut));

    ranktableInfo = virtTopoBuilder.GetRankTableInfo(); // 获取ranktable信息
    topoInfo = virtTopoBuilder.GetTopoInfo(); // 获取topo信息
    rankSize = newVirtualTopo->GetRankSize();
    CheckVirtualTopo();
}

void CommunicatorImpl::AppendLocalDieIdForLinks()
{
    auto srcRankNode = newVirtualTopo->GetPeer(myRank)->GetNodeId();

    // 枚举level
    for(auto level : newVirtualTopo->GetLevels(myRank))
    {
        auto netInstance = newVirtualTopo->GetNetInstanceByRankId(level, myRank);
        // 每个连向Fabric的link
        auto& vGraph = netInstance->GetGraph();
        auto fabrics = netInstance->GetFabrics();
        for(auto fabric : fabrics)
        {
            auto dstRankNode = fabric->GetNodeId();
            auto links = vGraph.GetEdges(srcRankNode, dstRankNode);
            for(auto link : links)
            {
                auto dieId = GetLocalDieId(link->GetSourceIface()->GetAddr());
                link->GetSourceIface()->SetLocalDieId(dieId);
            }
        }
        // 直接连向对端rank的link
        for(u32 dstRank = 0; dstRank < rankSize; dstRank++)
        {
            auto dstRankNode = newVirtualTopo->GetPeer(dstRank)->GetNodeId();
            auto links = vGraph.GetEdges(srcRankNode, dstRankNode);
            for(auto link : links)
            {
                auto dieId = GetLocalDieId(link->GetSourceIface()->GetAddr());
                link->GetSourceIface()->SetLocalDieId(dieId);
            }
        }
    }
}

void CommunicatorImpl::InitDataBufferManager()
{
    // 申请scratchMem
    u64 scratchBufSize = static_cast<u64>(GetBufferSize());
    if(scratchBufSize == 0) {
        scratchBufSize = EnvConfig::GetInstance().GetAlgoConfig().GetBuffSize();
    } else {
        scratchBufSize = scratchBufSize * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE;
    }
    cclBufferSize = scratchBufSize;
    HCCL_DEBUG("[CommunicatorImpl][InitDataBufferManager] scratchBufSize[%llu]", scratchBufSize);
    cclBuffer = std::make_shared<DevBuffer>(scratchBufSize);
}

void CommunicatorImpl::InitCollService()
{
    collService = std::make_unique<CollServiceStub>(this);
    collService->Init();
}

RankId CommunicatorImpl::GetMyRank() const
{
    return myRank;
}

u32 CommunicatorImpl::GetRankSize() const
{
    return rankSize;
}

u64 CommunicatorImpl::GetBufferSize() const
{
    return cclBufferSize;
}

void CommunicatorImpl::InitFeatureFlag()
{
    // ccuFeatureFlag = EnvConfig::GetInstance().GetFeatureSwitchConfig().GetCcuEnable();
    // HCCL_DEBUG("ccuFeatureFlag [%u]", ccuFeatureFlag);

    // auto hcclOpMode = EnvConfig::GetInstance().GetAlgoConfig().GetOpExpansionMode();
    // if (hcclOpMode == OpExpansionMode::MIX) {
    //     devModeFlag = true;
    //     HCCL_DEBUG("devModeFlag [%d]", devModeFlag);
    // }
}

void CommunicatorImpl::InitCcuInstance()
{
    if (rankSize == 1) {
        HCCL_INFO("CommunicatorImpl[%s] RankSize 1 no need initCcuInstance", __func__);
        return;
    }
    CcuComponent::GetInstance(devLogicId).Init();
    CcuResBatchAllocator::GetInstance(devLogicId).Init();
    CtxMgrImp::GetInstance(devLogicId).Init();
}

bool CommunicatorImpl::GetCcuFeatureFlag() const
{
    return ccuFeatureFlag != 0;
}

CollServiceStub *CommunicatorImpl::GetCollService() const
{
    return collService.get();
}

CommunicatorImpl::~CommunicatorImpl()
{
    HCCL_DEBUG("[~CommunicatorImpl]start CommunicatorImpl destory");
}

bool CommunicatorImpl::IsOpUsingCcuMs() const
{
    return true;
}

bool CommunicatorImpl::IsOpUsingCcuSched() const
{
    return true;
}

bool CommunicatorImpl::IsCommUsingCcuMs() const
{
    return true;
}

bool CommunicatorImpl::IsCommUsingCcuSched() const
{
    return true;
}

HcclResult CommunicatorImpl::ReLoadCollOp()
{
    HcclResult ret = HCCL_SUCCESS;
    return ret;
}

void CommunicatorImpl::AcceleratorFallback()
{
    
}

bool CommunicatorImpl::IfAccStateConfigExplicitly() const
{
    return false;
}
} // namespace Hccl
