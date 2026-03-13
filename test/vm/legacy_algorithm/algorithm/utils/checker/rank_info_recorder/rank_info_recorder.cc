#include "rank_info_recorder.h"

namespace checker {
RankInfoRecorder* RankInfoRecorder::Global()
{
    static RankInfoRecorder* rankInfoRecorder = new RankInfoRecorder;
    return rankInfoRecorder;
}

void RankInfoRecorder::Reset()
{
    rankId2phyId.clear();
    rankId2serverId.clear();
    rankId2superpodId.clear();
    return;
}

void RankInfoRecorder::SetRankId(RankId rankId)
{
    curRankId = rankId;
    return;
}

RankId RankInfoRecorder::GetRankId()
{
    return curRankId;
}

void RankInfoRecorder::SetDevType(CheckerDevType devType)
{
    curDevType = devType;
    return;
}

CheckerDevType RankInfoRecorder::GetDevType()
{
    return curDevType;
}

u32 RankInfoRecorder::GetRankSize()
{
    return rankSize_;
}

void RankInfoRecorder::InitRankInfo(TopoMeta topoMeta, CheckerDevType uniDevType)
{
    u32 myRankId = 0;
    for (int i = 0; i < topoMeta.size(); i++) {
        for (int j = 0; j < topoMeta[i].size(); j++) {
            for (int k = 0; k < topoMeta[i][j].size(); k++) {
                rankId2phyId[myRankId] = topoMeta[i][j][k];
                rankId2serverId[myRankId] = j;
                rankId2superpodId[myRankId] = i;
                myRankId++;
            }
        }
    }
    rankSize_ = myRankId;
}

}