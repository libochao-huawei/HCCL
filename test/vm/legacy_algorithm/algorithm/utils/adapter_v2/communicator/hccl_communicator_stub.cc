#include "hccl_communicator_stub.h"

#include <memory>
#include <utility>
#include "communicator_impl_stub.h"
#include "log.h"

namespace Hccl {

std::string g_algNamev2;

HcclCommunicator::HcclCommunicator(const CommParams &commParams) : commParams(std::move(commParams))
{
    pimpl                    = std::make_unique<CommunicatorImpl>();
    config.hcclBufferSize    = 0;
    config.hcclDeterministic = 0;
}

HcclCommunicator::HcclCommunicator(const CommParams &commParams, const HcclCommConfig *config)
    : commParams(std::move(commParams)), config(*config)
{
    pimpl = std::make_unique<CommunicatorImpl>();
}

HcclCommunicator::~HcclCommunicator()
{
}

HcclResult HcclCommunicator::Init(const std::string &ranktableM, std::string& topoPath)
{
    return pimpl->Init(commParams, ranktableM, topoPath, config);
}

CommunicatorImpl *HcclCommunicator::GetCommImpl()
{
    return pimpl.get();
}

void HcclCommunicator::DeInit() const
{
}

HcclResult HcclCommunicator::LoadOpbasedCollOp(CollOpParams &opParams, std::string& algName)
{
    g_algNamev2 = algName;
    return pimpl->LoadOpbasedCollOp(opParams, algName);
}

HcclResult HcclCommunicator::LoadOffloadCollOp(std::string &opTag, CollOpParams &opParams)
{
    return pimpl->LoadOffloadCollOp(opTag, opParams);
}

HcclResult HcclCommunicator::GetRankSize(uint32_t *rankSize)
{
    if (rankSize == nullptr) {
        HCCL_ERROR("Parameter rank size is nullptr.");
        return HcclResult::HCCL_E_PARA;
    }

    *rankSize = pimpl->GetRankSize();

    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRankId(uint32_t &rankId)
{
    rankId = pimpl->GetMyRank();
    return HcclResult::HCCL_SUCCESS;
}

CcuInsPreprocessor* HcclCommunicator::GetCcuInsPreprocessor()
{
    return pimpl->GetCcuInsPreprocessor();
}

void HcclCommunicator::TransformTask()
{
    return pimpl->TransformTask();
}

} // namespace Hccl
