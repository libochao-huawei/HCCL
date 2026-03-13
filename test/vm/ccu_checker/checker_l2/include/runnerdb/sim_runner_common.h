#ifndef _SIM_RUNNER_COMMOM_H_
#define _SIM_RUNNER_COMMOM_H_

#include <thread>
#include "acl/acl_base.h"
#include "sim_models.h"
#include "sim_runner_db.h"

namespace sim {

aclError GetDeviceByLogicId(uint32_t deviceId, sim::Device &device);
aclError GetDeviceByPhysicalId(uint32_t deviceId, sim::Device &device);
aclError UpdateDeviceLogicId(uint64_t serverKey, uint32_t phyDevId, uint32_t logicDevId);
aclError GetCcuFromDeviceByDieId(uint64_t deviceKey, uint8_t dieId, sim::Ccu &ccu);
aclError GetPortFromSpecCcuByName(uint64_t ccuKey, const std::string& portName, sim::Port &port);
aclError GetCcuResourceByCcu(uint64_t ccuKey, sim::CcuResource &ccuRes);
aclError GetContextByDevId(uint32_t deviceId, sim::Context &context);
aclError GetPortByName(uint64_t serverKey, uint32_t phyDevId, const std::string &name, sim::Port &port);
aclError GetPortByEid(const std::array<uint8_t, URMA_EID_LEN> &eid, sim::Port &port);
aclError GetPortByIpAddr(const std::string &ip, sim::Port &port);
aclError GetPortByCtxHandle(uint64_t ctxHandle, sim::Port &port);
aclError GetEndPointPairByDstPort(uint64_t portKey, sim::EndPointPair &endPointPair);

uint32_t GetAICpuCount(uint64_t deviceId);
uint32_t GetAICoreCount(uint64_t deviceId);
uint32_t GetVectorCoreCount(uint64_t deviceId);
}
#endif // _SIM_RUNNER_COMMOM_H_