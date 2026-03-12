#ifndef HCCLV1_RANK_INFO_RECORDER_H
#define HCCLV1_RANK_INFO_RECORDER_H
#include <map>
#include "llt_common.h"
#include "topo_meta.h"
#include "checker_def.h"

#ifndef HCCL_ALG_ANALYZER_DAVID
using namespace hccl;
#endif

namespace checker {

class RankInfoRecorder {
public:
    static RankInfoRecorder* Global();
    void Reset();

    void SetRankId(RankId rankId);
    RankId GetRankId();
    void SetDevType(CheckerDevType devType);
    CheckerDevType GetDevType();
    u32 GetRankSize();

    void InitRankInfo(TopoMeta topoMeta, CheckerDevType uniDevType);

    RankId curRankId = 0;
    CheckerDevType curDevType = CheckerDevType::DEV_TYPE_NOSOC;

    std::map<u32, u32> rankId2phyId;
    std::map<u32, u32> rankId2serverId;
    std::map<u32, u32> rankId2superpodId;
    u32 rankSize_ = 0;
};

}

#endif