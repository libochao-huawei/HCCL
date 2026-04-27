#ifndef HCCL_EVENT_16_WRAPPER_H_
#define HCCL_EVENT_16_WRAPPER_H_

#include <vector>
#include "ccu_kernel.h"

namespace ops_hccl {

constexpr uint32_t BITS_PER_CKE = 16;

class Event16Wrapper {
public:
    Event16Wrapper() : ckeCount_(0), rankSize_(0), kernel_(nullptr) {}

    HcclResult Init(hcomm::CcuKernel* kernel, uint32_t rankSize);

    void SetMask(uint32_t mask);

    void RecordEvent();

    HcclResult WaitEvent();

    HcclResult WriteNb(ChannelHandle channel, const hcomm::CcuRep::RemoteAddr& dst,
                       const hcomm::CcuRep::LocalAddr& src, const hcomm::CcuRep::Variable& len);

    HcclResult WriteNb(ChannelHandle channel, const hcomm::CcuRep::RemoteAddr& dst,
                       const hcomm::CcuRep::CcuBuf& src, const hcomm::CcuRep::Variable& len);

    uint32_t GetCkeCount() const { return ckeCount_; }

private:
    HcclResult WaitEventInternal(hcomm::CcuRep::CompletedEvent& event);

    std::vector<hcomm::CcuRep::CompletedEvent> events_;
    uint32_t ckeCount_;
    uint32_t rankSize_;
    hcomm::CcuKernel* kernel_;
    uint32_t currentMask_{0};
};

}  // namespace ops_hccl

#endif  // HCCL_EVENT_16_WRAPPER_H_