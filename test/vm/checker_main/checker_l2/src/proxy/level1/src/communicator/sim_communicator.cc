#include "sim_communicator.h"

using namespace std;
using namespace HcclSim;

namespace HcclProxy {

HcclResult Sim_HcclCommInitClusterInfo(ShmCommDomain *commDomain, uint32_t rank, HcclComm *comm)
{
    SimCommunicator* communicator = new SimCommunicator();
    // CHK_RET(communicator->Init(topoMeta, rank));
    auto ret = communicator->Init(commDomain, rank, commDomain->rankNum);
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [%s] communicator init fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    *comm = static_cast<HcclComm>(communicator);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimCommunicator::Init(const char *clusterInfo, uint32_t rank)
{
    printf("[ERROR] [SimCommunicator::%s] not support", __func__);
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult SimCommunicator::Init(ShmCommDomain *commDomain, uint32_t rank, uint32_t rankSize)
{
    curRank_ = rank;
    rankSize_ = rankSize;

    // 生成rankGraphs
    // std::vector<GraphRankInfo> rankGraphs;
    // // CHK_PRT(GenGraphRankInfos(topoMeta, rankGraphs));
    // auto ret = GenGraphRankInfos(commDomain, rankGraphs);
    // if (ret != HcclResult::HCCL_SUCCESS) {
    //     printf("[ERROR] [SimCommunicator::%s] GenGraphRankInfos fail", __func__);
    //     return HcclResult::HCCL_E_PARA;
    // }

    topoModel_ = unique_ptr<TopoModel>(new TopoModel());
    // HCCL_DEBUG("[SimCommunicator::%s] rankSize[%u], ", __func__, topoModel_->GetRankSize());
    auto ret = topoModel_->Init();
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] topoModel_ init fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    printf("[INFO] [SimCommunicator::%s] topoModel_ init success, rankSize[%u], ", __func__, topoModel_->GetRankSize());

    // 获取默认commConfig
    HcclCommConfig comConfig;
    // CHK_PRT(GetDefaultCommConfig(comConfig, "hccl_world_group"));
    ret = GetDefaultCommConfig(comConfig, "hccl_world_group");
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] GetDefaultCommConfig fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    identifier_ = comConfig.hcclCommName;
    // CHK_PRT(SetIndependentOpConfig(comConfig));
    ret = SetIndependentOpConfig(comConfig);
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] SetIndependentOpConfig fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    // manager初始化
    contextManager_ = unique_ptr<SimContextMgr>(new SimContextMgr());
    channelMgr_ = unique_ptr<SimChannelMgr>(new SimChannelMgr(commId_, curRank_, rankSize_));

    // CCL buffer 初始化
    auto cclRet = AllocNpuMemory(curRank_, cclBufferSize_, &cclBufferAddr_);
    if (cclRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] Alloc ccl buffer fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    ShmSimNpu* curNpu = nullptr;
    cclRet = GetNpuByRankId(curRank_, &curNpu);
    if (cclRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] get curRank npu fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    curNpu->cclIdx = (curNpu->memCount - 1);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimCommunicator::MockInit(ShmCommDomain *commDomain, uint32_t rank, uint32_t rankSize)
{
    curRank_ = rank;
    rankSize_ = rankSize;

    topoModel_ = unique_ptr<TopoModel>(new TopoModel());
    // HCCL_DEBUG("[SimCommunicator::%s] rankSize[%u], ", __func__, topoModel_->GetRankSize());
    auto ret = topoModel_->Init();
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] topoModel_ init fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    printf("[INFO] [SimCommunicator::%s] topoModel_ init success, rankSize[%u], ", __func__, topoModel_->GetRankSize());

    // 获取默认commConfig
    HcclCommConfig comConfig;
    // CHK_PRT(GetDefaultCommConfig(comConfig, "hccl_world_group"));
    ret = GetDefaultCommConfig(comConfig, "hccl_world_group");
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] GetDefaultCommConfig fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    identifier_ = comConfig.hcclCommName;
    // CHK_PRT(SetIndependentOpConfig(comConfig));
    ret = SetIndependentOpConfig(comConfig);
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] SetIndependentOpConfig fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    // manager初始化
    contextManager_ = unique_ptr<SimContextMgr>(new SimContextMgr());
    channelMgr_ = unique_ptr<SimChannelMgr>(new SimChannelMgr(commId_, curRank_, rankSize_));

    // Mock CCL buffer 初始化
    void* cclMockAddr{0};
    auto cclRet = MockAllocNpuMemory(curRank_, cclBufferSize_, &cclMockAddr);
    if (cclRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] Alloc ccl buffer fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    cclBufferAddr_ = cclMockAddr;
    ShmSimNpu* curNpu = nullptr;
    cclRet = GetNpuByRankId(curRank_, &curNpu);
    if (cclRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] get curRank npu fail", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    cclRet = RegisterNpuMemory(curRank_, cclMockAddr, cclBufferSize_, 2);
    if (cclRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimCommunicator::%s] RegisterNpuMemory", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    curNpu->cclIdx = (curNpu->memCount - 1);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimCommunicator::SetIndependentOpConfig(const HcclCommConfig &commConfig)
{
    commId_ = commConfig.hcclCommName;
    // HCCL_INFO("[%s] commEngine[%d], threadNum[%u], notifyNumPerThread[%u], commId[%s]",
    //     __func__, commEngine_, threadNum_, notifyNumPerThread_, commId_.c_str());
    printf("[INFO] [SimCommunicator::%s] commEngine[%d], threadNum[%u], notifyNumPerThread[%u], commId[%s]",
        __func__, commEngine_, threadNum_, notifyNumPerThread_, commId_.c_str());
    
    if (!independentOpThreadMgr_) {
        independentOpThreadMgr_ = unique_ptr<SimThreadMgr>(new SimThreadMgr(commId_, curRank_));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimCommunicator::GetDefaultCommConfig(HcclCommConfig &commConfig, const std::string &commName) const
{
    commConfig.hcclBufferSize = 1024; // GetExternalInputCCLBuffSize();
    commConfig.hcclDeterministic = 1024; // GetExternalInputHcclDeterministicV2();
    auto ret = strncpy_s(commConfig.hcclCommName, ROOTINFO_INDENTIFIER_MAX_LENGTH, commName.c_str(), commName.size());
    if (ret != EOK) {
        // HCCL_ERROR("[%s] str copy fail. return %d", __func__, ret);
        printf("[ERROR] [SimCommunicator::%s] str copy fail. return %d", __func__, ret);
        return HcclResult::HCCL_E_INTERNAL;
    }
    commConfig.hcclOpExpansionMode = 0;
    commConfig.hcclRdmaTrafficClass = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET;
    commConfig.hcclRdmaServiceLevel = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET;
    commConfig.hcclWorldRankID  = 0;
    commConfig.hcclJobID  = 0;
    return HcclResult::HCCL_SUCCESS;
}

uint32_t SimCommunicator::GetRankId()
{
    return curRank_;
}

uint32_t SimCommunicator::GetRankSize()
{
    if (topoModel_ == nullptr) {
        return 0;
    }
    return topoModel_->GetRankSize();
}

std::string SimCommunicator::GetIdentifier()
{
    return identifier_;
}

HcclResult SimCommunicator::GetCommRankGraph(void **graph, uint32_t *len)
{
    *graph = topoModel_->rankGraphs_.data();
    *len = topoModel_->rankGraphs_.size() * sizeof(GraphRankInfo);
    // HCCL_INFO("[%s] len[%u], rankSize[%u], sizeof(GraphRankInfo)[%u]",
    //     __func__, *len, topoModel_->rankGraphs_.size(), sizeof(GraphRankInfo));
    printf("[INFO] [SimCommunicator::%s] len[%u], rankSize[%lu], sizeof(GraphRankInfo)[%lu]",
        __func__, *len, topoModel_->rankGraphs_.size(), sizeof(GraphRankInfo));
    
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimCommunicator::GetHcclBuffer(CommBuffer *buffer)
{
    // CHK_PTR_NULL(buffer);
    if (buffer == nullptr) {
        printf("[ERROR] [SimCommunicator::%s] buffer is NULL", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    buffer->addr = cclBufferAddr_;
    buffer->size = cclBufferSize_;
    
    return HcclResult::HCCL_SUCCESS;
}

}