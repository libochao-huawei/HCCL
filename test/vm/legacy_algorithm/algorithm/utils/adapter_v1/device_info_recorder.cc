#include "device_info_recorder.h"
#include "transformer.h"
namespace hccl {

DeviceInfoRecorder* DeviceInfoRecorder::Global()
{
    static DeviceInfoRecorder* rankInfoRecorder = new DeviceInfoRecorder;
    return rankInfoRecorder;
}

void DeviceInfoRecorder::Reset()
{
    rankId2devType.clear();
    rankId2superdeviceId.clear();
    return;
}

void DeviceInfoRecorder::InitDeviceInfo(TopoMeta topoMeta, RankTable_t &rankTable, CheckerDevType uniDevType)
{
    u32 myRankId = 0;
    for (int i = 0; i < topoMeta.size(); i++) {
        for (int j = 0; j < topoMeta[i].size(); j++) {
            for (int k = 0; k < topoMeta[i][j].size(); k++) {
                if (g_HcclDevType2CheckerDevType[rankTable.rankList[myRankId].deviceInfo.deviceType] != CheckerDevType::DEV_TYPE_NOSOC) {
                    rankId2devType[myRankId] = g_HcclDevType2CheckerDevType[rankTable.rankList[myRankId].deviceInfo.deviceType];
                } else {
                    rankId2devType[myRankId] = uniDevType;
                }

                rankId2superdeviceId[myRankId] = rankTable.rankList[myRankId].superDeviceId;
                myRankId++;
            }
        }
    }
}

}