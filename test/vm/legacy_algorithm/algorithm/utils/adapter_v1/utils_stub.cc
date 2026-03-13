#include "utils_stub.h"

namespace hccl {

HcclResult CheckCurRankId(RankId curRank, RankId srcRank, RankId dstRank)
{
    if (curRank != srcRank && curRank != dstRank) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

}

