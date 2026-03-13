#ifndef HCCLV1_DEVICE_INFO_RECORDER_H
#define HCCLV1_DEVICE_INFO_RECORDER_H

#include "llt_common.h"
#include "topo_meta.h"
#include "checker_def.h"
using namespace checker;

namespace hccl {

class DeviceInfoRecorder {
public:
    static DeviceInfoRecorder* Global();
    void Reset();

    void InitDeviceInfo(TopoMeta topoMeta, RankTable_t &rankTable, CheckerDevType uniDevType);
    std::map<u32, CheckerDevType> rankId2devType;
    std::map<u32, u32> rankId2superdeviceId;
};

}

#endif