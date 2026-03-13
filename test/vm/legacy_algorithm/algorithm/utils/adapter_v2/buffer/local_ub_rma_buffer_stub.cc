
#include "local_ub_rma_buffer.h"

namespace Hccl {

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle)
    : LocalRmaBuffer(buf, RmaType::UB), rdmaHandle(rdmaHandle)
{
    ;
}

LocalUbRmaBuffer::~LocalUbRmaBuffer()
{
    ;
}

string LocalUbRmaBuffer::Describe() const
{
    return "";
}

std::unique_ptr<Serializable> LocalUbRmaBuffer::GetExchangeDto()
{
    return std::unique_ptr<Serializable>(nullptr);
}

u32 LocalUbRmaBuffer::GetTokenId() const
{
    return tokenId;
}

u32 LocalUbRmaBuffer::GetTokenValue() const
{
    return tokenValue;
}

TokenIdHandle LocalUbRmaBuffer::GetTokenIdHandle() const
{
    return tokenIdHandle;
}

u32 GetUbToken()
{
    constexpr uint32_t tokenValue = 1;
    return tokenValue;
}

}