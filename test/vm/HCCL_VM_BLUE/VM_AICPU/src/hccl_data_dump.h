#ifndef HCCL_DATA_DUMP_H
#define HCCL_DATA_DUMP_H

#include "hccl_common_defs.h"
#include <string>

using namespace HcclSim;

HcclVmResult DumpData();

std::string GenDataId();

HcclVmResult DumpModel(std::string dataId);
HcclVmResult DumpMemLayout(std::string dataId);
HcclVmResult DumpTask(std::string dataId);

HcclVmResult DumpHcclVmSynthesisData(const std::string &dataId);
HcclVmResult DumpHcclVmInstrData(const std::string &dataId);
HcclVmResult DumpHcclVmTask(const std::string &dataId);

#endif