#include <sstream>
#include "sim_channel.h"

using namespace std;
using namespace HcclSim;

namespace HcclProxy {

SimChannel::SimChannel(const std::string& commId, const std::string& tag, CommEngine engine, CommProtocol protocol,
    uint32_t locRankId, uint32_t rmtRankId, uint32_t notifyNum)
    : commId_(commId), tag_(tag), engine_(engine), protocol_(protocol), locRankId_(locRankId), rmtRankId_(rmtRankId), notifyNum_(notifyNum)
{}

SimChannel::~SimChannel()
{
    auto ret = Destrory();
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[ERROR] [SimChannel] 析构失败 \n");
    }
}

HcclResult SimChannel::Init()
{
    ShmSimNpu* locNpu = nullptr;
    auto ret = GetNpuByRankId(locRankId_, &locNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimChannel::%s] get npu fail \n", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    if (locNpu->cclIdx != MAX_MEM_BLOCK_NUM) {
        locMemIdx_ = locNpu->cclIdx;
    } else {
        auto ret = AllocNpuMemoryGetIdx(locRankId_, memSize, &locMemIdx_);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [SimChannel::%s] AllocNpuMemoryGetIdx fail \n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
        locNpu->cclIdx = locMemIdx_;
    }

    for (uint32_t i = 0; i < notifyNum_; ++i) {
        void* notifyPtr = nullptr;
        auto ret = AllocNotify(locRankId_, &notifyPtr);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [SimChannel::%s] AllocNotify fail \n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
        uint64_t notifyId = reinterpret_cast<ShmSimNotify*>(notifyPtr)->notifyId.value;
        locNotifys_.push_back(notifyId);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SimChannel::Destrory() {
    ShmSimNpu* locNpu = nullptr;
    auto retLoc = GetNpuByRankId(locRankId_, &locNpu);
    if (retLoc != HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [SimChannel::%s] get npu fail\n", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    locNpu->cclIdx = MAX_MEM_BLOCK_NUM;
    
    for (auto& notifyId : locNotifys_) {
        auto ret = ReleaseNotify(notifyId);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [SimChannel::%s] loc release notify fail \n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

string SimChannel::ToString() const
{
    stringstream ss;
    ss << "Channel" << "{" << to_string(locRankId_) << "->" << to_string(rmtRankId_) << "}";
    return ss.str();
}

// LinkProtoStub SimChannel::GetLinkType() const
// {
//     // 根据业务代码 HcclCommunicator::BuildChannelRequests()
//     return (protocol_ == CommProtocol::COMM_PROTOCOL_ROCE) ? LinkProtoStub::RDMA : LinkProtoStub::SDMA;
// }

uint64_t SimChannel::GetLocNotifyIdByIndex(uint32_t notifyIdx) const
{
    if (notifyIdx >= locNotifys_.size()) {
        THROW<InvalidParamsException>("[SimChannel::%s] notifyIndex[%u] out of bound", __func__, notifyIdx);
    }
    return locNotifys_[notifyIdx];
}

uint64_t SimChannel::GetRmtNotifyIdByIndex(uint32_t notifyIdx) const
{
    if (notifyIdx >= rmtNotifys_.size()) {
        THROW<InvalidParamsException>("[SimChannel::%s] notifyIndex[%u] out of bound", __func__, notifyIdx);
    }
    return rmtNotifys_[notifyIdx];
}

void SimChannel::SetRmtNotifys(uint64_t* notifyIds, size_t nums) {
    for (int i = 0; i < nums; ++i) {
        rmtNotifys_.push_back(notifyIds[i]);
    }
}

}   // namespace HcclSim