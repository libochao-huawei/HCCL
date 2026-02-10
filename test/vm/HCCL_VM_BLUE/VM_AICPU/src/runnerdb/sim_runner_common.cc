#include <iostream>
#include <runnerdb/sim_runner_common.h>
#include "hccl_vm_log.h"

namespace sim {

aclError GetDeviceByLogicId(uint32_t deviceId, sim::Device &device)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find device by logical id {:d}", __func__, deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = ret.first;
    return ACL_SUCCESS;
}

aclError GetDeviceByPhysicalId(uint32_t deviceId, sim::Device &device)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.physical_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find device by physical id {:d}", __func__, deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = ret.first;
    return ACL_SUCCESS;
}

aclError UpdateDeviceLogicId(uint32_t phyDevId, uint32_t logicDevId)
{
    sim::Device device{};
    auto ret = GetDeviceByPhysicalId(phyDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    RunnerDB::Update<sim::Device>(deviceKey, [deviceKey, logicDevId](sim::Device &dev) { 
        dev.logic_id = logicDevId;
    });

    return ACL_SUCCESS;
}

aclError UpdateSuperDeviceId(uint32_t logicDevId, uint32_t superDeviceId)
{
    sim::Device device{};
    auto ret = GetDeviceByLogicId(logicDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    RunnerDB::Update<sim::Device>(deviceKey, [deviceKey, superDeviceId](sim::Device &dev) { 
        dev.super_device_id = superDeviceId;
    });

    return ACL_SUCCESS;
}

aclError GetCcuFromDeviceByDieId(uint64_t deviceKey, uint8_t dieId, sim::Ccu &ccu)
{
    auto ret = RunnerDB::GetOneByPred<sim::Ccu>([deviceKey, dieId](const sim::Ccu &c) {
        return c.device_id == deviceKey && c.die_id == dieId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find ccu from device {:d} by die {:d}", __func__, deviceKey, static_cast<uint32_t>(dieId));
        return ACL_ERROR_INVALID_PARAM;
    }

    ccu = ret.first;
    return ACL_SUCCESS;
}

aclError GetCcuResourceByCcu(uint64_t ccuKey, sim::CcuResource &ccuRes)
{
    auto ret = RunnerDB::GetOneByPred<sim::CcuResource>([ccuKey](const sim::CcuResource &cr) {
        return cr.ccu_id == ccuKey;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find ccu resource from ccu {:d}", __func__, ccuKey);
        return ACL_ERROR_INVALID_PARAM;
    }

    ccuRes = ret.first;
    return ACL_SUCCESS;
}

aclError GetContextByDevId(uint32_t deviceId, sim::Context &context)
{
    auto ret = RunnerDB::GetOneByPred<sim::Context>([deviceId](const sim::Context &ctx) {
        return ctx.device_id == deviceId && ctx.is_default == 1;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find context by logical device id {:d}", __func__, deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    context = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortByName(uint32_t phyDevId, const std::string &name, sim::Port &port)
{
    sim::Device device{};
    auto ret1 = GetDeviceByPhysicalId(phyDevId, device);
    if (ret1 != ACL_SUCCESS) {
        HCCL_VM_ERROR("[{}] get device by physical device id({:d}) failed", __func__, phyDevId);
        return ret1;
    }

    auto deviceKey = device.id;
    auto ret2 = RunnerDB::GetOneByPred<sim::Port>([deviceKey, name](const sim::Port &p) {
        return deviceKey == p.device_id && strcmp(p.name, name.c_str()) == 0;
    });
    if (!ret2.second) {
        HCCL_VM_ERROR("[{}] cannot find port by name", __func__, name.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret2.first;
    return ACL_SUCCESS;
}

aclError GetPortByEid(const std::array<uint8_t, URMA_EID_LEN> &eid, sim::Port &port)
{
    // auto ret = RunnerDB::GetOneByPred<sim::Port>([eid](const sim::Port &p) {
    //     return std::equal(eid.begin(), eid.end(), std::begin(p.eid));
    // });
    // if (!ret.second) {
    //     printf("[ERROR][GetPortByEid] cannot find port by eid\n");
    //     return ACL_ERROR_INVALID_PARAM;
    // }

    // port = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortByIpAddr(const std::string &ip, sim::Port &port)
{
    auto ret = RunnerDB::GetOneByPred<sim::Port>([ip](const sim::Port &p) {
        std::cout<<"zhf-find ip..."<<ip<<", "<<p.ip_addr<<std::endl;
        return strcmp(p.ip_addr, ip.c_str()) == 0;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find port by ip addr:{}", __func__, ip.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortFromSpecCcuByName(uint64_t ccuKey, const std::string& portName, sim::Port &port)
{
    auto ret = RunnerDB::GetOneByPred<sim::Port>([ccuKey, portName](const sim::Port &p) {
        return p.ccu_id == ccuKey && strcmp(p.name, portName.c_str()) == 0;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find port by port name:{}", __func__, portName.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortByCtxHandle(uint64_t ctxHandle, sim::Port &port)
{
    auto ret = RunnerDB::GetOneByPred<sim::Port>([ctxHandle](const sim::Port &p) {
        return ctxHandle == p.rdma_handle;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find port by ctx_handle", __func__);
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret.first;
    return ACL_SUCCESS;
}

/*
    根据Port找对应EndpointPair：
    框内链路：src port和dst port都在一个EndpointPair中，入参portKey表示dst port key;
    出框链路：src port和dst port在不同的EndpointPiar中，
             EndpointPair中仅保存src port。入参portKey表示src port key；
    查找方案：
    1. 先按照dst port查找：
       1.1 若找到endpointPair，则isInServer=true表示框内链路；
       1.2 若找不到endpointPair，则isInServer=false表示出框链路；
    2. 若dst port找不到endpointPair，则再按照src port查找：
       2.1 若找到endpointPair，则
*/
aclError GetEndPointPairByDstPort(uint64_t portKey, sim::EndPointPair &endPointPair)
{
    auto ret = RunnerDB::GetOneByPred<sim::EndPointPair>([portKey](const sim::EndPointPair &ep) {
        return ep.src_port == portKey || ep.dst_port == portKey;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find end point pair by port: {:d}", __func__, portKey);
        return ACL_ERROR_INVALID_PARAM;
    }

    endPointPair = ret.first;
    return ACL_SUCCESS;
}

uint32_t GetAICpuCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_AICPU_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find device by physical id {:d}", __func__, deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto aiCpus = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_CPU;
    });

    return aiCpus.size();
}

uint32_t GetAICoreCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_AICORE_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find device by physical id {:d}", __func__, deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
    });

    return scalars.size();
}

uint32_t GetVectorCoreCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_VECTOR_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find device by physical id {:d}", __func__, deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
    });

    uint32_t vectorCoreCount = 0;
    for (auto scalar : scalars) {
        auto tsIdx = scalar.id;
        auto vectorCores = RunnerDB::GetByPred<sim::ComputeDie>([tsIdx](const sim::ComputeDie &die) {
            return die.ts_id == tsIdx && die.type == (uint8_t)COMPUTE_TYPE_VECTOR;
        });

        vectorCoreCount += vectorCores.size();
    }
    return vectorCoreCount;
}

uint32_t GetCubeCoreCount(uint64_t deviceId)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find device by physical id {:d}", __func__, deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
    });

    uint32_t vectorCoreCount = 0;
    for (auto scalar : scalars) {
        auto tsIdx = scalar.id;
        auto vectorCores = RunnerDB::GetByPred<sim::ComputeDie>([tsIdx](const sim::ComputeDie &die) {
            return die.ts_id == tsIdx && die.type == (uint8_t)COMPUTE_TYPE_CUBE;
        });

        vectorCoreCount += vectorCores.size();
    }
    return vectorCoreCount;
}

}
