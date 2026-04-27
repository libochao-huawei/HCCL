#include "event_16_wrapper.h"

namespace ops_hccl {

HcclResult Event16Wrapper::Init(hcomm::CcuKernel* kernel, uint32_t rankSize)
{
    rankSize_ = rankSize;
    ckeCount_ = (rankSize + BITS_PER_CKE - 1) / BITS_PER_CKE;
    kernel_ = kernel;

    events_.resize(ckeCount_);
    for (uint32_t i = 0; i < ckeCount_; i++) {
        events_[i] = kernel_->CreateCompletedEvent();
    }
    return HCCL_SUCCESS;
}

void Event16Wrapper::SetMask(uint32_t mask)
{
    currentMask_ = mask;
}

void Event16Wrapper::RecordEvent()
{
    if (ckeCount_ == 1) {
        events_[0].SetMask(currentMask_);
        kernel_->RecordEvent(events_[0]);
    } else {
        for (uint32_t i = 0; i < ckeCount_; i++) {
            uint32_t maskInCke = (currentMask_ >> (i * BITS_PER_CKE)) & ((1 << BITS_PER_CKE) - 1);
            if (maskInCke != 0) {
                events_[i].SetMask(maskInCke);
                kernel_->RecordEvent(events_[i]);
            }
        }
    }
}

HcclResult Event16Wrapper::WaitEvent()
{
    if (ckeCount_ == 1) {
        events_[0].SetMask(currentMask_);
        return kernel_->WaitEvent(events_[0]);
    } else {
        for (uint32_t i = 0; i < ckeCount_; i++) {
            uint32_t maskInCke = (currentMask_ >> (i * BITS_PER_CKE)) & ((1 << BITS_PER_CKE) - 1);
            if (maskInCke != 0) {
                events_[i].SetMask(maskInCke);
                CHK_RET(kernel_->WaitEvent(events_[i]));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult Event16Wrapper::WriteNb(ChannelHandle channel, const hcomm::CcuRep::RemoteAddr& dst,
                                    const hcomm::CcuRep::LocalAddr& src, const hcomm::CcuRep::Variable& len)
{
    if (ckeCount_ == 1) {
        events_[0].SetMask(currentMask_);
        return kernel_->WriteNb(channel, dst, src, len, events_[0]);
    } else {
        for (uint32_t i = 0; i < ckeCount_; i++) {
            uint32_t maskInCke = (currentMask_ >> (i * BITS_PER_CKE)) & ((1 << BITS_PER_CKE) - 1);
            if (maskInCke != 0) {
                events_[i].SetMask(maskInCke);
                CHK_RET(kernel_->WriteNb(channel, dst, src, len, events_[i]));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult Event16Wrapper::WriteNb(ChannelHandle channel, const hcomm::CcuRep::RemoteAddr& dst,
                                    const hcomm::CcuRep::CcuBuf& src, const hcomm::CcuRep::Variable& len)
{
    if (ckeCount_ == 1) {
        events_[0].SetMask(currentMask_);
        return kernel_->WriteNb(channel, dst, src, len, events_[0]);
    } else {
        for (uint32_t i = 0; i < ckeCount_; i++) {
            uint32_t maskInCke = (currentMask_ >> (i * BITS_PER_CKE)) & ((1 << BITS_PER_CKE) - 1);
            if (maskInCke != 0) {
                events_[i].SetMask(maskInCke);
                CHK_RET(kernel_->WriteNb(channel, dst, src, len, events_[i]));
            }
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl