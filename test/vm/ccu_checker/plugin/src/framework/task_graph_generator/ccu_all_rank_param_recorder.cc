#include "ccu_all_rank_param_recorder.h"
// #include "types.h"

namespace HcclSim {

AllRankParamRecorder* AllRankParamRecorder::Global()
{
    static AllRankParamRecorder* allRankParamRecorder = new AllRankParamRecorder;
    return allRankParamRecorder;
}

void AllRankParamRecorder::Reset()
{
    curXn.clear();
    curGSA.clear();
    curCKE.clear();
    seenPost.clear();
    return;
}

void AllRankParamRecorder::InitParam()
{
    return;
}

HcclResult AllRankParamRecorder::SetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t xnValue)
{
    // std::cout<<"[INFO][SetXn] Set xn, rankId= "<<rankId<<", dieId= "<<static_cast<int>(dieId)<<", xnId= "<<xnId<<", xnValue= "<<xnValue<<std::endl;
    curXn[rankId][dieId][xnId] = xnValue;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::SetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t gsaValue)
{
    curGSA[rankId][dieId][gsaId] = gsaValue;
    // std::cout<<"zhf-set gsa: rank= "<<rankId<<", die= "<<dieId<<", gsaId= "<<gsaId<<", value="<<gsaValue<<std::endl;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::SetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t ckeValue)
{
    curCKE[rankId][dieId][ckeId] = ckeValue;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::GetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue)
{
    if(curXn.find(rankId) != curXn.end()) {
        if (curXn[rankId].find(dieId) != curXn[rankId].end()) {
            if (curXn[rankId][dieId].find(xnId) != curXn[rankId][dieId].end()) {
                xnValue = curXn[rankId][dieId][xnId];
                // std::cout<<"zhf-get xn: rank= "<<rankId<<", die= "<<dieId<<", xnId= "<<xnId<<", value="<<std::hex<<xnValue<<std::endl;
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[GetXn]curXn is not be initialized, rankId[%u], dieId[%u], xnId[%hu]", rankId, dieId, xnId);
    return HCCL_E_PARA;
}

HcclResult  AllRankParamRecorder::GetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t& gsaValue)
{
    if(curGSA.find(rankId) != curGSA.end()) {
        if (curGSA[rankId].find(dieId) != curGSA[rankId].end()) {
            if (curGSA[rankId][dieId].find(gsaId) != curGSA[rankId][dieId].end()) {
                gsaValue = curGSA[rankId][dieId][gsaId];
                // std::cout<<"zhf-get gsa: rank= "<<rankId<<", die= "<<dieId<<", gsaId= "<<gsaId<<", value="<<std::hex<<gsaValue<<std::endl;
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[GetGSA]curGSA is not be initialized, rankId[%u], dieId[%u], gsaId[%hu]", rankId, dieId, gsaId);
    return HCCL_E_PARA;
}

HcclResult AllRankParamRecorder::GetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t& ckeValue)
{
    if(curCKE.find(rankId) != curCKE.end()) {
        if (curCKE[rankId].find(dieId) != curCKE[rankId].end()) {
            if (curCKE[rankId][dieId].find(ckeId) != curCKE[rankId][dieId].end()) {
                ckeValue = curCKE[rankId][dieId][ckeId];
                return HCCL_SUCCESS;
            }
        }
    }

    ckeValue = 0;
    return HCCL_SUCCESS;
}

HcclResult AllRankParamRecorder::CheckAllPostMatch()
{
    for (auto& rankPair : seenPost) {
        RankId rank = rankPair.first;
        for (auto& diePair : seenPost[rank]) {
            uint32_t dieId = diePair.first;
            for (auto& regPair: seenPost[rank][dieId]) {
                uint16_t regId = regPair.first;
                for (auto& post : seenPost[rank][dieId][regId]) {
                    HCCL_WARNING("unmatched LocalPost/Post: %s", post->task->Describe().c_str());
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

}