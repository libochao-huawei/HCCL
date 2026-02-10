/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <vector>
#include <atomic>
#include <iostream>
#include <securec.h>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "runtime/base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "hccl_common_macro.h"
#include "sim_runner_common.h"
#include "dtype_common.h"
#include "hccl_vm_log.h"

// current host id
uint64_t g_host_id;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus


int rtModelFake = 0;
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI)
{   
    *modelRI = &rtModelFake;
    return ACL_SUCCESS;
}

HcclResult hrtGetDeviceIndexByPhyId(uint32_t devicePhyId, uint32_t &deviceLogicId)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([devicePhyId](const sim::Device& d) {
        return d.physical_id  == (uint32_t)devicePhyId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[hrtGetDeviceIndexByPhyId] cannot find device by physical id {:d}", devicePhyId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    deviceLogicId = ret.first.logic_id;
    return HCCL_SUCCESS;
}

aclError aclrtSetDevice(int32_t deviceId)
{
    // sleep(30);
    HCCL_VM_DEBUG("[aclrtSetDevice] start set device {:d}", deviceId);
    auto runner = sim::GetCurrRunnerTls();
    auto curRunnerId = runner.id;

    // 三种可能：1. 第一次SetDevice；2. 重复SetDevice 3. 切换Device
    // 操作：要么新建context和stream，设置current context；要么获取已有context，设置current context。
    // 2. 根据logic device id查找Device
    sim::Device device{};
    auto devRet = sim::GetDeviceByLogicId((uint32_t)deviceId, device);
    if (devRet != ACL_SUCCESS) {
        return devRet;
    }

    uint64_t currCtxId = 0;
    uint64_t deviceKey = device.id;
    // 查找device对应context信息
    auto ret = RunnerDB::GetOneByPred<sim::Context>([deviceKey](const sim::Context &ctx) {
        return ctx.device_id == deviceKey && ctx.is_default == 1;
    });
    if (!ret.second) {
        // 找不到context，则默认新增一个context
        sim::Context context{};
        context.device_id = device.id;
        context.run_id = curRunnerId;
        context.is_default = 1;
        context.ref_cnt = 1;
        currCtxId = RunnerDB::Add<sim::Context>(context);

        // 新建context，默认新建一个stream
        sim::Stream stream{};
        stream.ctx_id = currCtxId;
        stream.activated = 1;
        stream.is_primary_default = 1;
        RunnerDB::Add<sim::Stream>(stream);
    } else {
        currCtxId = ret.first.id;
        RunnerDB::Update<sim::Context>(currCtxId, [](sim::Context &ctx) { ctx.ref_cnt++;});
        currCtxId = ret.first.id;
    }

    // runner表中，current context更新
    sim::SetCurrCtxTls(currCtxId);
    return ACL_SUCCESS;
}

aclError aclrtResetDevice(int32_t deviceId)
{
    std::cout<<"zhf-[aclrtResetDevice] deviceId= "<<deviceId<<std::endl;
    auto runner = sim::GetCurrRunnerTls();
    auto curCtxId = runner.current_ctx_id;
    if (curCtxId == 0) {
        return ACL_SUCCESS;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(curCtxId);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtResetDevice] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto streamRet = RunnerDB::GetOneByPred<sim::Stream>([curCtxId](const sim::Stream& stream) {
        return stream.ctx_id == curCtxId && stream.is_primary_default == 1;
    });
    if (!streamRet.second) {
        // not find
        HCCL_VM_ERROR("[aclrtResetDevice] can not find stream of context {:d}", curCtxId);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (currCtx->ref_cnt > 1) {
        RunnerDB::Update<sim::Context>(curCtxId, [](sim::Context &ctx) { ctx.ref_cnt--;});
    } else {
        // 删除stream
        RunnerDB::Delete<sim::Stream>(streamRet.first.id);
        // 删除context
        RunnerDB::Delete<sim::Context>(curCtxId);
    }

    // 设置runner中current context id无效
    // runner表中，current context更新
    curCtxId = 0;
    sim::SetCurrCtxTls(curCtxId);
    return ACL_SUCCESS;
}

aclError aclrtResetDeviceForce(int32_t deviceId)
{
    HCCL_VM_DEBUG("[aclrtResetDeviceForce] can not find stream of deviceId {:d}", deviceId);
    auto runner = sim::GetCurrRunnerTls();
    auto curCtxId = runner.current_ctx_id;
    if (curCtxId == 0) {
        return ACL_SUCCESS;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(curCtxId);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtResetDevice] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto streamRet = RunnerDB::GetOneByPred<sim::Stream>([curCtxId](const sim::Stream& stream) {
        return stream.ctx_id == curCtxId && stream.is_primary_default == 1;
    });
    if (!streamRet.second) {
        // not find
        HCCL_VM_ERROR("[aclrtResetDevice] can not find stream of context {:d}", curCtxId);
        return ACL_ERROR_INVALID_PARAM;
    }

    // 删除stream
    RunnerDB::Delete<sim::Stream>(streamRet.first.id);
    // 删除context
    RunnerDB::Delete<sim::Context>(curCtxId);
    // 设置runner中current context id无效
    // runner表中，current context更新
    curCtxId = 0;
    sim::SetCurrCtxTls(curCtxId);
    return ACL_SUCCESS;
}

aclError aclrtGetDevice(int32_t* device)
{
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtGetDevice] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!devRes.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtGetDevice] can not find current device id:{:d}", currCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    *device = devRes->logic_id;
    HCCL_VM_DEBUG("[aclstub][aclrtGetDevice]device: {:d}", devRes->logic_id);
    return ACL_SUCCESS;
}

aclError aclrtGetRunMode(aclrtRunMode *runMode)
{
    // 疑问， 模拟器是不是只有一种Mode
    *runMode = ACL_DEVICE;
    return ACL_SUCCESS;
}

aclError aclrtSetTsDevice(aclrtTsId tsId)
{
    if (tsId == ACL_TS_ID_AICORE) {
        sim::SetTsDevice(tsId);
    }

    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtSetTsDevice] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!devRes.has_value()) {
        HCCL_VM_ERROR("[aclrtSetTsDevice] can not find current device id:{:d}", currCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t vectoCount = sim::GetVectorCoreCount(devRes->logic_id);
    if (vectoCount != 0) {
        sim::SetTsDevice(tsId);
    }

    return ACL_SUCCESS;
}

aclError aclrtGetDeviceCount(uint32_t *count)
{
    auto runner = sim::GetCurrRunnerTls();
    auto devs = RunnerDB::GetByPred<sim::Device>([](const sim::Device& d) {
        return d.status == 0;
    });

    if (devs.empty()) {
        HCCL_VM_ERROR("[aclstub][aclrtGetDeviceCount]devices failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    *count = devs.size();
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[aclrtGetDeviceUtilizationRate] cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    utilizationInfo->cubeUtilization    = 20;
    utilizationInfo->vectorUtilization = 20;
    utilizationInfo->aicpuUtilization  = 20;
    utilizationInfo->memoryUtilization  = 20;
    return ACL_SUCCESS;
}

aclError aclrtQueryDeviceStatus(int32_t deviceId, aclrtDeviceStatus *deviceStatus)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.logic_id  == (uint32_t)deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[aclrtQueryDeviceStatus] cannot find device by logic id {:d}", deviceId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    *deviceStatus = (aclrtDeviceStatus)ret.first.status;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

const char *aclrtGetSocName()
{
    auto runner = sim::GetCurrRunnerTls();
    auto hostId = runner.host_id;
    if (hostId == 0) {
        // not find
        HCCL_VM_ERROR("[aclrtGetSocName] wrong host id: {:d}", hostId);
        return "invalid param";
    }

    auto host = RunnerDB::GetOneByPred<sim::Host>([hostId](const sim::Host& h) {
        return h.id == hostId;
    });
    if (!host.second) {
        // not find
        HCCL_VM_ERROR("[aclrtResetDevice] can not find host by key {:d}", hostId);
        return "";
    }

    auto serverId = host.first.server_id;
    auto devRes = RunnerDB::GetOneByPred<sim::Device>([serverId](const sim::Device& d) {
        return d.server_id == serverId;
    });
    if (!devRes.second) {
        // not find
        HCCL_VM_ERROR("[aclrtResetDevice] can not find device by server id {:d}", serverId);
        return "";
    }

    thread_local static char SocName[128] = {0};
    memcpy(SocName, devRes.first.soc_version, strlen(devRes.first.soc_version));
    SocName[strlen(devRes.first.soc_version)] = '\0';
    HCCL_VM_DEBUG("[aclstub][aclrtGetSocName]device: {}", devRes.first.soc_version);
    return SocName;
}

aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode)
{
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtGetDevice] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto curDevId = currCtx->device_id;
    RunnerDB::Update<sim::Device>(curDevId, [curDevId, mode](sim::Device &dev) { dev.overflow_mode = mode; });
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode *mode)
{
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtGetDeviceSatMode] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtGetDeviceSatMode] can not find current device:{:d}", currCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    *mode = (aclrtFloatOverflowMode)dev->overflow_mode;
    return ACL_SUCCESS;
}

aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId)
{
    auto dev1 = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!dev1.second) {
        HCCL_VM_ERROR("[aclrtDeviceCanAccessPeer] cannot find device by logic id {:d}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev2 = RunnerDB::GetOneByPred<sim::Device>([peerDeviceId](const sim::Device &d) {
        return d.logic_id == peerDeviceId;
    });
    if (!dev2.second) {
        HCCL_VM_ERROR("[aclrtDeviceCanAccessPeer] cannot find device by logic id {:d}", peerDeviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev1Id = dev1.first.id;
    auto dev2Id = dev2.first.id;

    auto ret = RunnerDB::GetOneByPred<sim::DeviceConnection>([dev1Id, dev2Id](const sim::DeviceConnection& devConn) {
        return devConn.src_dev_id  == dev1Id && devConn.dst_dev_id  == dev2Id;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[aclrtDeviceCanAccessPeer] get device connection failed srcdev:{:d} dstdev:{:d}", dev1Id, dev2Id);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    *canAccessPeer = (int32_t)ret.first.access_by_remote;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags)
{
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev1 = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev1.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] can not find current device id:{:d}", currCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (dev1->logic_id == peerDeviceId) {
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] invalid device id:{:d}", peerDeviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev2 = RunnerDB::GetOneByPred<sim::Device>([peerDeviceId](const sim::Device &d) {
        return d.logic_id == peerDeviceId;
    });
    if (!dev2.second) {
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] cannot find device by logic id {:d}", peerDeviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev1Id = dev1->id;
    auto dev2Id = dev2.first.id;
    // 查找connection 记录
    auto ret = RunnerDB::GetOneByPred<sim::DeviceConnection>([dev1Id, dev2Id](const sim::DeviceConnection& devConn) {
        return devConn.src_dev_id  == dev1Id && devConn.dst_dev_id  == dev2Id;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[aclrtDeviceCanAccessPeer] get device connection failed srcdev:{:d} dstdev:{:d}", dev1Id, dev2Id);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    // 更新数据库
    RunnerDB::Update<sim::DeviceConnection>(ret.first.id, [](sim::DeviceConnection &devConn) { devConn.access_by_remote = 1;});
    return ACL_SUCCESS;
}

aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId)
{
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev1 = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev1.has_value()) {
        // not find
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] can not find current device id:{:d}", currCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (dev1->logic_id == peerDeviceId) {
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] invalid device id:{:d}", peerDeviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev2 = RunnerDB::GetOneByPred<sim::Device>([peerDeviceId](const sim::Device &d) {
        return d.logic_id == peerDeviceId;
    });
    if (!dev2.second) {
        HCCL_VM_ERROR("[aclrtDeviceEnablePeerAccess] cannot find device by logic id {:d}", peerDeviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev1Id = dev1->id;
    auto dev2Id = dev2.first.id;
    // 查找connection 记录
    auto ret = RunnerDB::GetOneByPred<sim::DeviceConnection>([dev1Id, dev2Id](const sim::DeviceConnection& devConn) {
        return devConn.src_dev_id  == dev1Id && devConn.dst_dev_id  == dev2Id;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[aclrtDeviceCanAccessPeer] get device connection failed srcdev:{:d} dstdev:{:d}", dev1Id, dev2Id);
        return ACL_ERROR_INVALID_PARAM;
    }

    // 更新数据库
    RunnerDB::Update<sim::DeviceConnection>(ret.first.id, [](sim::DeviceConnection &devConn) { devConn.access_by_remote = 0;});
    return ACL_SUCCESS;
}

aclError aclrtGetOverflowStatus(void *outputAddr, size_t outputSize, aclrtStream stream)
{
    uint64_t streamIdx = (uint64_t)(uintptr_t)stream;
    auto stmRes = RunnerDB::GetById<sim::Stream>(streamIdx);
    if (!stmRes.has_value()) {
        HCCL_VM_ERROR("can not get stream:{:d}", streamIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto ctxRes = RunnerDB::GetById<sim::Context>(stmRes->ctx_id);
    if (!ctxRes.has_value()) {
        HCCL_VM_ERROR("can not get context:{:d}", stmRes->ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceIdx = ctxRes->device_id;
    auto devStatusRes = RunnerDB::GetOneByPred<sim::DeviceStatus>([deviceIdx](const sim::DeviceStatus& dev) {
        return dev.device_id  == deviceIdx;
    });
    if (!devStatusRes.second) {
        HCCL_VM_ERROR("[aclrtGetOverflowStatus] get device :{:d} failed", deviceIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint8_t* tmp = (uint8_t *)outputAddr;
    *tmp = devStatusRes.first.overflow_status;

    return ACL_SUCCESS;
}

aclError aclrtResetOverflowStatus(aclrtStream stream)
{
    uint64_t streamIdx = (uint64_t)(uintptr_t)stream;
    auto stmRes = RunnerDB::GetById<sim::Stream>(streamIdx);
    if (!stmRes.has_value()) {
        HCCL_VM_ERROR("can not get stream:{:d}", streamIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto ctxRes = RunnerDB::GetById<sim::Context>(stmRes->ctx_id);
    if (!ctxRes.has_value()) {
        HCCL_VM_ERROR("can not get Context:{:d}", stmRes->ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceIdx = ctxRes->device_id;
    auto devStatusRes = RunnerDB::GetOneByPred<sim::DeviceStatus>(
        [deviceIdx](const sim::DeviceStatus &dev) { return dev.device_id == deviceIdx; });
    if (!devStatusRes.second) {
        HCCL_VM_ERROR("[aclrtResetOverflowStatus] get device :{:d} failed", deviceIdx);
        return ACL_ERROR_INVALID_PARAM;
    }
    RunnerDB::Update<sim::DeviceStatus>(devStatusRes.first.id, [](sim::DeviceStatus &devStatus) { devStatus.overflow_status = 0;});
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeDevice(void)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeDeviceWithTimeout(int32_t timeout)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value)
{
    uint32_t count = 0;
    if (attr == ACL_DEV_ATTR_AICPU_CORE_NUM) {
        count = sim::GetAICpuCount(deviceId);
    } else if (attr == ACL_DEV_ATTR_AICORE_CORE_NUM) {
        count = sim::GetAICoreCount(deviceId);
    } else if (attr == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
        count = sim::GetVectorCoreCount(deviceId);
    }
    *value = (int64_t)count;
    return ACL_SUCCESS;
}

aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceCapability(int32_t deviceId, aclrtDevFeatureType devFeatureType, int32_t *value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetDevicesTopo(uint32_t deviceId, uint32_t otherDeviceId, uint64_t *value)
{
    auto dev1 = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
        return d.logic_id == deviceId;
    });
    if (!dev1.second) {
        HCCL_VM_ERROR("[aclrtGetDevicesTopo] cannot find device by logic id {:d}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev2 = RunnerDB::GetOneByPred<sim::Device>([otherDeviceId](const sim::Device &d) {
        return d.logic_id == otherDeviceId;
    });
    if (!dev2.second) {
        HCCL_VM_ERROR("[aclrtGetDevicesTopo] cannot find device by logic id {:d}", otherDeviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev1Id = dev1.first.id;
    auto dev2Id = dev2.first.id;

    auto ret = RunnerDB::GetOneByPred<sim::DeviceConnection>([dev1Id, dev2Id](const sim::DeviceConnection& devConn) {
        return devConn.src_dev_id  == dev1Id && devConn.dst_dev_id  == dev2Id;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[aclrtGetDevicesTopo] get device connection failed srcdev:{:d} dstdev:{:d}", dev1Id, dev2Id);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    *value = (uint64_t)ret.first.link_type;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

aclError aclrtDevicePeerAccessStatus(int32_t deviceId, int32_t peerDeviceId, int32_t *status)
{
    return aclrtDeviceCanAccessPeer(status, deviceId, peerDeviceId);
}

aclError aclInit(const char *configPath)
{
    HCCL_VM_DEBUG("[{0}] Success", __func__);
    return ACL_SUCCESS;
}

aclError aclFinalize()
{
    HCCL_VM_DEBUG("[{0}] Success", __func__);
    return ACL_SUCCESS;
}

////////////////////////////////////////////////
aclError aclrtGetPhyDevIdByLogicDevId(int32_t logicDevId, int32_t *const phyDevId)
{
    HCCL_VM_DEBUG("[aclstub][aclrtGetPhyDevIdByLogicDevId]Success");
    sim::Device device{};
    auto devRet = sim::GetDeviceByLogicId((uint32_t)logicDevId, device);
    if (devRet != ACL_SUCCESS) {
        return devRet;
    }

    *phyDevId = (int32_t)device.physical_id;
    return ACL_SUCCESS;
}

aclError aclrtGetLogicDevIdByPhyDevId(const int32_t phyDevId, int32_t *const logicDevId)
{
    HCCL_VM_DEBUG("[aclstub][aclrtGetLogicDevIdByPhyDevId]Success");
    sim::Device device{};
    auto devRet = sim::GetDeviceByPhysicalId((uint32_t)phyDevId, device);
    if (devRet != ACL_SUCCESS) {
        return devRet;
    }

    *logicDevId = (int32_t)device.logic_id;
    return ACL_SUCCESS;
}

aclError aclrtSetDeviceTaskAbortCallback(const char *regName, aclrtDeviceTaskAbortCallback callback, void *args)
{
    // TODO
    return ACL_SUCCESS;
}

////////////////////rt接口/////////////////////
rtError_t rtGetDevicePhyIdByIndex(uint32_t devIndex, uint32_t *phyId)
{
    sim::Device device{};
    auto devRet = sim::GetDeviceByLogicId((uint32_t)devIndex, device);
    if (devRet != ACL_SUCCESS) {
        return devRet;
    }
    *phyId = device.physical_id;
    return ACL_SUCCESS;
}

rtError_t rtGetPhyDeviceInfo(uint32_t phyId, int32_t moduleType, int32_t infoType, int64_t *val)
{
    // TODO
    return ACL_SUCCESS;
}

rtError_t rtGetDeviceIndexByPhyId(uint32_t phyId, uint32_t *devIndex)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([phyId](const sim::Device& d) {
        return d.physical_id  == phyId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[rtGetDeviceIndexByPhyId] cannot find device by physical id {:d}", phyId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    *devIndex = ret.first.logic_id;
    return ACL_SUCCESS;
}

rtError_t rtSetDevice(int32_t devId)
{
    return aclrtSetDevice(devId);
}

rtError_t rtGetPairPhyDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val)
{
    HCCL_VM_DEBUG("[aclstub][rtGetPairPhyDevicesInfo]Success");
    *val = 1;
    return ACL_SUCCESS;
}

rtError_t rtsGetLogicDevIdByPhyDevId(int32_t phyDevId, int32_t * const logicDevId)
{
    return aclrtGetLogicDevIdByPhyDevId(phyDevId, logicDevId);
}

struct rtDevResInfo;
rtError_t rtReleaseDevResAddress(rtDevResInfo * const resInfo)
{
    HCCL_VM_DEBUG("[aclstub][rtReleaseDevResAddress]Success");
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus