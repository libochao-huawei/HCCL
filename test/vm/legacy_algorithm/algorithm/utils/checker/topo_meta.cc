#include "topo_meta.h"
#include "topoinfo_struct.h"
#include "hccl_types.h"
#include <string>
#include <arpa/inet.h>
#include <vector>
#include <algorithm>

using namespace hccl;

namespace checker {

HcclResult RankTable_For_LLT::GenTopoMeta(TopoMeta &topoMate, int superPodNum, int serverNum, int rankNum)
{
    for (u32 i = 0; i < superPodNum; i++) {  // box
        SuperPodMeta superPodMeta;
        for (u32 j = 0; j < serverNum; j++) {  // serverNumPerBox
            ServerMeta serverMate;
            for (u32 k = 0; k < rankNum; k++) {
                serverMate.push_back(k);
            }
            superPodMeta.push_back(serverMate);
        }
        topoMate.push_back(superPodMeta);
    }
    return HCCL_SUCCESS;
}

u32 GetRankNumFormTopoMeta(TopoMeta &topoMeta)
{
    u32 rankNum = 0;
    for (auto &podMeta : topoMeta) {
        for (auto &serverMeta : podMeta) {
            rankNum += serverMeta.size();
        }
    }
    return rankNum;
}

u32 GetServerNumFormTopoMeta(TopoMeta &topoMeta)
{
    u32 sererNum = 0;
    for (auto &podMeta : topoMeta) {
        for (auto &serverMeta : podMeta) {
            if (serverMeta.size()) {
                sererNum++;
            }
        }
    }
    return sererNum;
}

}  // namespace checker
