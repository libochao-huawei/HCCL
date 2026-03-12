#ifndef HCCLV2_CCU_ALL_RANK_PARAM_RECORD_H
#define HCCLV2_CCU_ALL_RANK_PARAM_RECORD_H

#include <map>
#include <set>
#include <vector>
#include <hccl/hccl_types.h>
#include "base.h"
#include "log.h"
#include "task_graph_generator.h"

namespace Hccl {

class AllRankParamRecorder {
public:
    static AllRankParamRecorder* Global();
    void InitParam();
    void Reset();
    HcclResult CheckAllPostMatch();

    HcclResult SetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t xnValue);
    HcclResult SetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t gsaValue);
    HcclResult SetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t ckeValue);

    HcclResult GetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue);
    HcclResult GetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t& gsaValue);
    HcclResult GetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t& ckeValue);

    // rankId -> dieId -> 寄存器Id -> 寄存器value
    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint64_t>>> curXn;
    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint64_t>>> curGSA;
    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint16_t>>> curCKE;

    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, std::set<checker::TaskNode*>>>> seenPost;
};


}

#endif