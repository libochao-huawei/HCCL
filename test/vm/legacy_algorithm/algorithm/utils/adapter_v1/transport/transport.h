#ifndef TESTS_STUB_TRANSPORT_H
#define TESTS_STUB_TRANSPORT_H

#include "transport_pub.h"
#include "coll_alg_param.h"

namespace hccl {
class TransportCompared {
public:
    void Describe() {
        HCCL_DEBUG("TODO");
    };
    u32 GetTransportId() {
        return transportId_;
    }
    HcclResult SetTransportId(u32 transportId) {
        transportId_ = transportId;
        return HCCL_SUCCESS;
    }

    RankId localRank = 0;
    RankId remoteRank = 0;
    bool isValid = false;
    u32 transportId_;
    bool isCompared = false;
    TransportMemType inputMemType = TransportMemType::RESERVED;
    TransportMemType outputMemType = TransportMemType::RESERVED;
    TransportMemType remoteinputMemType = TransportMemType::RESERVED;
    TransportMemType remoteoutputMemType = TransportMemType::RESERVED;
};
}  // namespace checker

#endif