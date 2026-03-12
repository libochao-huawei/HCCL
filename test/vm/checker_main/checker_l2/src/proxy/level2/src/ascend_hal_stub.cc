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
#include <chrono>
#include <ctime>
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "ascend_hal.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "ascend_hal_mdc.h"
#include "hccl_vm_log.h"


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

drvError_t halDeviceOpen(uint32_t devid, halDevOpenIn *in, halDevOpenOut *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halDeviceClose(uint32_t devid, halDevCloseIn *in)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halProcessResBackup(halProcResBackupInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halProcessResRestore(halProcResRestoreInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceStateNotifierRegister(drvDeviceStateNotify state_callback)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceStartupRegister(drvDeviceStartupNotify startup_callback)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetChipCapability(uint32_t devId, struct halCapabilityInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetCapabilityGroupInfo(int device_id, int ts_id, int group_id, struct capability_group_info *group_info,
                                     int group_count)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetAPIVersion(int *halAPIVersion)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

void halSetRuntimeApiVer(int Version)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
}

drvError_t drvDeviceStatus(uint32_t devId, drvStatus_t *status)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceOpen(void **devInfo, uint32_t devId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceClose(uint32_t devId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceGetTransWay(void *src, void *dest, uint8_t *trans_type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvGetPlatformInfo(uint32_t *info)
{
    *info = 1;
    printf("[HAL][STUB][drvGetPlatformInfo]info:%u\n", *info);
    return DRV_ERROR_NONE;
}

drvError_t drvGetDevNum(uint32_t *num_dev)
{
    auto runner = sim::GetCurrRunnerTls();
    auto devs = RunnerDB::GetByPred<sim::Device>([](const sim::Device& d) {
        return d.status == 0;
    });

    if (devs.empty()) {
        printf("[HAL][STUB][drvGetDevNum]devices failed\n");
        return DRV_ERROR_NO_DEVICE;
    }
    *num_dev = devs.size();
    printf("[HAL][STUB][drvGetDevNum]num_dev:%u\n", *num_dev);
    return DRV_ERROR_NONE;
}

drvError_t drvGetDevIDByLocalDevID(uint32_t localDevId, uint32_t *devId)
{
    *devId = localDevId;
    printf("[HAL][STUB][drvGetDevIDByLocalDevID] devId:%u, localDevId:%u\n", localDevId, *devId);
    return DRV_ERROR_NONE;
}

drvError_t halGetDevProbeList(uint32_t *devices, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvGetDevIDs(uint32_t *devices, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvGetDeviceLocalIDs(uint32_t *devices, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvGetLocalDevIDByHostDevID(uint32_t host_dev_id, uint32_t *local_dev_id)
{
    local_dev_id = &host_dev_id;
    printf("[HAL][STUB][drvGetLocalDevIDByHostDevID] host_dev_id:%u, local_dev_id:%u\n", host_dev_id, *local_dev_id);
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeRegister(uint32_t devId, struct halSensorNodeCfg *cfg, uint64_t *handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUnregister(uint32_t devId, uint64_t handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUpdateState(uint32_t devId, uint64_t handle, int val, halGeneralEventType_t assertion)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    if (moduleType == (int32_t)MODULE_TYPE_SYSTEM && infoType == (int32_t)INFO_TYPE_VERSION) {
        int64_t hardwareVersion = 0xf00;
    } else if ((moduleType == (int32_t)MODULE_TYPE_SYSTEM) && (infoType == (int32_t)INFO_TYPE_CORE_NUM)) {
        *value = 1;
    } else {
        *value = 0;
    }
    // *value = hardwareVersion;
    return DRV_ERROR_NONE;
}

drvError_t halGetSocVersion(uint32_t devId, char *soc_version, uint32_t len)
{
    strcpy(soc_version, "Ascend950");
    return DRV_ERROR_NONE;
}

drvError_t halGetDeviceInfoByBuff(uint32_t devId, int32_t moduleType, int32_t infoType, void *buf, int32_t *size)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSetDeviceInfoByBuff(uint32_t devId, int32_t moduleType, int32_t infoType, void *buf, int32_t size)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetPhyDeviceInfo(uint32_t phyId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *value)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetPairPhyDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *value)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvCustomCall(uint32_t devId, uint32_t cmd, void *para)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceExceptionHookRegister(drvDeviceExceptionReporFunc exception_callback_func)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

void drvFlushCache(uint64_t base, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return;
}

drvError_t drvDeviceGetPhyIdByIndex(uint32_t devIndex, uint32_t *phyId)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([devIndex](const sim::Device& d) {
        return d.logic_id  == devIndex;
    });
    if (!ret.second) {
        printf("[ERROR][drvDeviceGetPhyIdByIndex] cannot find device by devIndex id %u\n", devIndex);
        return DRV_ERROR_NO_DEVICE;
    }
    *phyId = (uint32_t)ret.first.physical_id;
    printf("[HAL][STUB][drvDeviceGetPhyIdByIndex] index:%u, phyId:%u\n", devIndex, *phyId);
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceGetIndexByPhyId(uint32_t phyId, uint32_t *devIndex)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([phyId](const sim::Device& d) {
        return d.physical_id  == phyId;
    });
    if (!ret.second) {
        printf("[ERROR][drvDeviceGetIndexByPhyId] cannot find device by phsical id %u\n", phyId);
        return DRV_ERROR_NO_DEVICE;
    }
    *devIndex = (uint32_t)ret.first.logic_id;
    printf("[HAL][STUB][drvDeviceGetIndexByPhyId] phyId:%u, devIndex:%u\n", phyId, *devIndex);
    return DRV_ERROR_NONE;
}

drvError_t drvGetProcessSign(struct process_sign *sign)
{
	memset(sign, 0, sizeof(struct process_sign));
    printf("[HAL][STUB][drvGetProcessSign]\n");
    return DRV_ERROR_NONE;
}

drvError_t halQueryDevpid(struct halQueryDevpidInfo info, pid_t *dev_pid) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvBindHostPid(struct drvBindHostpidInfo info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvUnbindHostPid(struct drvBindHostpidInfo info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvQueryProcessHostPid(int pid, unsigned int *chip_id, unsigned int *vfid, unsigned int *host_pid,
                                  unsigned int *cp_type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResAddrMap(unsigned int devId, struct res_addr_info *res_info, unsigned long *va, unsigned int *len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResAddrUnmap(unsigned int devId, struct res_addr_info *res_info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halBindCgroup(BIND_CGROUP_TYPE bindType)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

pid_t drvDeviceGetBareTgid(void)
{
    pid_t pid = getpid();
    printf("[HAL][STUB][drvDeviceGetBareTgid] pid:%u\n", pid);
    return pid;
}

drvError_t drvMemRead(uint32_t devId, MEM_CTRL_TYPE memType, uint32_t offset, uint8_t *value, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvMemWrite(uint32_t devId, MEM_CTRL_TYPE memType, uint32_t offset, uint8_t *value, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halDeviceEnableP2P(uint32_t dev, uint32_t peer_dev, uint32_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halDeviceDisableP2P(uint32_t dev, uint32_t peer_dev, uint32_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvGetP2PStatus(uint32_t dev, uint32_t peer_dev, uint32_t *status)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halDeviceCanAccessPeer(int *can_access_peer, uint32_t dev, uint32_t peer_dev)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvGetDeviceBootStatus(int phy_id, uint32_t *boot_status)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvNotifyIdAddrOffset(uint32_t devId, struct drvNotifyInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvCreateIpcNotify(struct drvIpcNotifyInfo *info, char *name, unsigned int len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvDestroyIpcNotify(const char *name, struct drvIpcNotifyInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvOpenIpcNotify(const char *name, struct drvIpcNotifyInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvCloseIpcNotify(const char *name, struct drvIpcNotifyInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvSetIpcNotifyPid(const char *name, pid_t pid[], int num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvRecordIpcNotify(const char *name)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdCreate(struct drvShrIdInfo *info, char *name, uint32_t name_len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdDestroy(const char *name)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdOpen(const char *name, struct drvShrIdInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdClose(const char *name)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdSetAttribute(const char *name, enum shrIdAttrType type, struct shrIdAttr attr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdGetAttribute(const char *name, enum shrIdAttrType type, struct shrIdAttr *attr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdInfoGet(const char *name, struct shrIdGetInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdSetPid(const char *name, pid_t pid[], uint32_t pid_num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdSetPodPid(const char *name, uint32_t sdid, pid_t pid)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halShrIdRecord(const char *name)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

void drvDfxShowReport(uint32_t devId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
}

drvError_t halSafeIslandTimeSyncMsgSend(uint32_t devId, void *msg, size_t msgSize)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMbindHbm(DVdeviceptr devPtr, size_t len, uint32_t type, uint32_t dev_id)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvLoadProgram(DVdevice device_id, void *program, unsigned int offset, size_t byte_count, void **v_ptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemAddressTranslate(DVdeviceptr vptr, UINT64 *pptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemSmmuQuery(DVdevice device, UINT32 *SSID)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemAllocL2buffAddr(DVdevice device, void **l2buff, UINT64 *pte)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemReleaseL2buffAddr(DVdevice device, void *l2buff)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemsetD8(DVdeviceptr dst, size_t destMax, UINT8 value, size_t num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemcpy(DVdeviceptr dst, size_t dest_max, DVdeviceptr src, size_t byte_count)
{
    //printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemcpy(void *dst, size_t dst_size, void *src, size_t count, struct memcpy_info *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemCpyAsync(DVdeviceptr dst, size_t dest_max, DVdeviceptr src, size_t byte_count,
                        uint64_t *copy_fd) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemCpyAsyncWaitFinish(uint64_t copy_fd) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemcpy2D(struct MEMCPY2D *p_copy) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemcpyBatch(uint64_t dst[], uint64_t src[], size_t size[], size_t count)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSdmaCopy(DVdeviceptr dst, size_t dst_size, DVdeviceptr src, size_t len) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSdmaBatchCopy(void *dst[], void *src[], size_t size[], int count) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemConvertAddr(DVdeviceptr p_src, DVdeviceptr p_dst, UINT32 len, struct DMA_ADDR *dma_addr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemDestroyAddr(struct DMA_ADDR *ptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemDestroyAddrBatch(struct DMA_ADDR *ptr[], uint32_t num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemcpySumbit(struct DMA_ADDR *dma_addr, int flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemcpyWait(struct DMA_ADDR *dma_addr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemPrefetchToDevice(DVdeviceptr dev_ptr, size_t len, DVdevice device)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemCreateHandle(DVdeviceptr vptr, size_t byte_count, char *name, uint32_t name_len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemDestroyHandle(const char *name)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemSetPidHandle(const char *name, int pid[], int num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemSetPodPid(const char *name, uint32_t sdid, int pid[], int num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemOpenHandle(const char *name, DVdeviceptr *vptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemOpenHandleByDevId(DVdevice dev_id, const char *name, DVdeviceptr *vptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemCloseHandle(DVdeviceptr vptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemSetAttribute(const char *name, uint32_t type, uint64_t attr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemGetAttribute(const char *name, enum ShmemAttrType type, uint64_t *attr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halShmemInfoGet(const char *name, struct ShmemGetInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult drvMemGetAttribute(DVdeviceptr vptr, struct DVattribute *attr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

int devmm_daemon_setup(void)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}
drvError_t halMemAgentOpen(uint32_t devid, uint32_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemAgentClose(uint32_t devid, uint32_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

int drvMemDeviceOpen(uint32_t devid, int devfd)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int drvMemDeviceClose(uint32_t devid)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

DVresult drvMemAllocProgram(void **pp, size_t bytesize)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halHostRegister(void *src_ptr, UINT64 size, UINT32 flag, UINT32 devid, void **dst_ptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halHostUnregister(void *src_ptr, UINT32 devid)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halHostUnregisterEx(void *src_ptr, UINT32 devid, UINT32 flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemFree(void *pp)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemAdvise(DVdeviceptr ptr, size_t count, unsigned int type, DVdevice device)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCheckProcessStatus(DVdevice device, processType_t process_type, processStatus_t status, bool *is_matched)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCheckProcessStatusEx(DVdevice device, processType_t process_type, processStatus_t status,
                                   struct drv_process_status_output *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemGetAddressReserveRange(void **ptr, size_t *size, drv_mem_addr_reserve_type type, uint64_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemAddressReserve(void **ptr, size_t size, size_t alignment, void *addr, uint64_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemAddressFree(void *ptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemCreate(drv_mem_handle_t **handle, size_t size, const struct drv_mem_prop *prop, uint64_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemRelease(drv_mem_handle_t *handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemMap(void *ptr, size_t size, size_t offset, drv_mem_handle_t *handle, uint64_t flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemUnmap(void *ptr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemSetAccess(void *ptr, size_t size, struct drv_mem_access_desc *desc, size_t count)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemGetAccess(void *ptr, struct drv_mem_location *location, uint64_t *flags)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemExportToShareableHandle(drv_mem_handle_t *handle, drv_mem_handle_type handle_type, uint64_t flags,
                                         uint64_t *shareable_handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemExportToShareableHandleV2(drv_mem_handle_t *handle, drv_mem_handle_type handle_type, uint64_t flags,
                                           struct MemShareHandle *share_handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemImportFromShareableHandle(uint64_t shareable_handle, uint32_t devid, drv_mem_handle_t **handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemImportFromShareableHandleV2(drv_mem_handle_type handle_type, struct MemShareHandle *share_handle,
                                             uint32_t devid, drv_mem_handle_t **handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemTransShareableHandle(drv_mem_handle_type handle_type, struct MemShareHandle *share_handle,
                                      uint32_t *server_id, uint64_t *shareable_handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemSetPidToShareableHandle(uint64_t shareable_handle, int pid[], uint32_t pid_num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemShareHandleSetAttribute(uint64_t shareable_handle, enum ShareHandleAttrType type,
                                         struct ShareHandleAttr attr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemShareHandleGetAttribute(uint64_t shareable_handle, enum ShareHandleAttrType type,
                                         struct ShareHandleAttr *attr)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemShareHandleInfoGet(uint64_t shareable_handle, struct ShareHandleGetInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemGetAllocationGranularity(const struct drv_mem_prop *prop, drv_mem_granularity_options option,
                                          size_t *granularity)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemGetInfoEx(DVdevice device, unsigned int type, struct MemInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemGetInfo(DVdevice device, unsigned int type, struct MemInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halMemCtl(int type, void *param_value, size_t param_value_size, void *out_value,
                     size_t *out_size_ret) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcGetCapacity(struct drvHdcCapacity *capacity)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcClientCreate(HDC_CLIENT *client, int maxSessionNum, int serviceType, int flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcClientDestroy(HDC_CLIENT client)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcSessionConnect(int peer_node, int peer_devid, HDC_CLIENT client, HDC_SESSION *session)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcSessionConnectEx(int peer_node, int peer_devid, int peer_pid, HDC_CLIENT client, HDC_SESSION *pSession)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcServerCreate(int devid, int serviceType, HDC_SERVER *pServer)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcServerDestroy(HDC_SERVER server)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcSessionAccept(HDC_SERVER server, HDC_SESSION *session)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcSessionClose(HDC_SESSION session)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcAllocMsg(HDC_SESSION session, struct drvHdcMsg **ppMsg, int count)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcFreeMsg(struct drvHdcMsg *msg)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcReuseMsg(struct drvHdcMsg *msg)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcAddMsgBuffer(struct drvHdcMsg *msg, char *pBuf, int len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcGetMsgBuffer(struct drvHdcMsg *msg, int index, char **pBuf, int *pLen)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcRecvPeek(HDC_SESSION session, int *msgLen, int flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcRecvBuf(HDC_SESSION session, char *pBuf, int bufLen, int *msgLen)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcSetSessionReference(HDC_SESSION session)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcGetTrustedBasePath(int peer_node, int peer_devid, char *base_path, unsigned int path_len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcSendFile(int peer_node, int peer_devid, const char *file, const char *dst_path,
                          void (*progress_notifier)(struct drvHdcProgInfo *))
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

void *drvHdcMallocEx(enum drvHdcMemType mem_type, void *addr, unsigned int align, unsigned int len, int devid,
                     unsigned int flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return nullptr;
}

drvError_t drvHdcFreeEx(enum drvHdcMemType mem_type, void *buf)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcDmaMap(enum drvHdcMemType mem_type, void *buf, int devid)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcDmaUnMap(enum drvHdcMemType mem_type, void *buf)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcDmaReMap(enum drvHdcMemType mem_type, void *buf, int devid)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

/* hdc epoll func */

drvError_t drvHdcEpollCreate(int size, HDC_EPOLL *epoll)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcEpollClose(HDC_EPOLL epoll)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcEpollCtl(HDC_EPOLL epoll, int op, void *target, struct drvHdcEvent *event)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t drvHdcEpollWait(HDC_EPOLL epoll, struct drvHdcEvent *events, int maxevents, int timeout, int *eventnum)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halHdcGetSessionInfo(HDC_SESSION session, struct drvHdcSessionInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcSend(HDC_SESSION session, struct drvHdcMsg *pMsg, UINT64 flag, UINT32 timeout)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcFastSend(HDC_SESSION session, struct drvHdcFastSendMsg msg, UINT64 flag, UINT32 timeout)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcRecv(HDC_SESSION session, struct drvHdcMsg *pMsg, int bufLen, UINT64 flag, int *recvBufCount,
                      UINT32 timeout)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcRecvEx(HDC_SESSION session, struct drvHdcMsg *pMsg, int bufLen, int *recvBufCount,
                        struct drvHdcRecvConfig *userConfig)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcFastRecv(HDC_SESSION session, struct drvHdcFastRecvMsg *msg, UINT64 flag, UINT32 timeout)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halHdcGetSessionAttr(HDC_SESSION session, int attr, int *value)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcGetServerAttr(HDC_SERVER server, int attr, int *value)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

hdcError_t halHdcNotifyRegister(int service_type, struct HdcSessionNotify *notify)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

void halHdcNotifyUnregister(int service_type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
}

hdcError_t halHdcSessionCloseEx(HDC_SESSION session, int type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

int dsmi_cmd_get_network_device_info(int device_id, const char *inbuf, unsigned int size_in, char *outbuf,
                                     unsigned int *size_out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int log_read_by_type(int device_id, char *buf, unsigned int *size, int timeout, enum log_channel_type channel_type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int log_set_level(int device_id, int channel_type, unsigned int log_level)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int log_get_channel_type(int device_id, int *channel_type_set, int *channel_type_num, int set_size)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int log_read(int device_id, char *buf, unsigned int *size, int timeout)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

// int log_set_dfx_param(uint32_t, uint32_t, void*, uint32_t)
// {
//     printf("[STUB][%s]stub is empty.\n", __func__);
//     return 0;
// }

// int log_get_dfx_param(uint32_t devid, uint32_t chan_type, void *data, uint32_t data_len)
// {
//     printf("[STUB][%s]stub is empty.\n", __func__);
//     return 0;
// }

void *log_type_alloc_mem(uint32_t device_id, uint32_t type, uint32_t *size)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return nullptr;
}
int prof_drv_get_channels(unsigned int device_id, channel_list_t *channels)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int prof_drv_start(unsigned int device_id, unsigned int channel_id, struct prof_start_para *start_para)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int prof_stop(unsigned int device_id, unsigned int channel_id)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int prof_channel_read(unsigned int device_id, unsigned int channel_id, char *out_buf, unsigned int buf_size)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int prof_channel_poll(struct prof_poll_info *out_buf, int num, int timeout)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halProfDataFlush(unsigned int device_id, unsigned int channel_id, unsigned int *data_len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halBuffAlloc(uint64_t size, void **buff)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

drvError_t halBuffGet(Mbuf *mbuf, void *buf, unsigned long size)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}
void halBuffPut(Mbuf *mbuf, void *buf)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
}
int halBuffPoolGet(void *poolStart)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halBuffPoolPut(void *poolStart)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halMbufAllocEx(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halMbufVerify(Mbuf *mbuf, unsigned int type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halMbufBuild(void *buff, uint64_t len, Mbuf **mbuf)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halMbufUnBuild(Mbuf *mbuf, void **buff, uint64_t *len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

drvError_t halBufEventSubscribe(const char *grpName, unsigned int threadGrpId, unsigned int event_id,
                                unsigned int devid) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halBufEventReport(const char *grpName)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGrpCacheAlloc(const char *name, unsigned int devId, GrpCacheAllocPara *para)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGrpCacheFree(const char *name, unsigned int devId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

int halGrpQuery(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff,
                unsigned int *outLen) ;

int halGrpQuery(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff,
                unsigned int *outLen) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

drvError_t halBuffProcCacheFree(unsigned long flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halBuffGetDQSPooInfo(struct mempool_t *mp, DqsPoolInfo *poolInfo)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halBuffGetDQSPooInfoById(unsigned int poolId, DqsPoolInfo *poolInfo)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halBuffCreateInterGrp(unsigned int grpId, const char *name, unsigned int len) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halBuffDestoryInterGrp(unsigned int grpId) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetMbufhandleforEnque(Mbuf *mbuf, uint64_t *handle)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdAlloc(uint32_t devId, struct halResourceIdInputInfo *in, struct halResourceIdOutputInfo *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdFree(uint32_t devId, struct halResourceIdInputInfo *in)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceEnable(uint32_t devId, struct halResourceIdInputInfo *in) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceDisable(uint32_t devId, struct halResourceIdInputInfo *in) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceConfig(uint32_t devId, struct halResourceIdInputInfo *in, struct halResourceConfigInfo *para)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceDetailQuery(uint32_t devId, struct halResourceIdInputInfo *in, struct halResourceDetailInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdCheck(struct drvResIdKey *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdInfoGet(struct drvResIdKey *key, drvResIdProcType type, uint64_t *value)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdRestore(struct drvResIdKey *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halTsdrvCtl(uint32_t devId, int cmd, void *param, size_t paramSize, void *out, size_t *outSize)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqCqAllocate(uint32_t devId, struct halSqCqInputInfo *in, struct halSqCqOutputInfo *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqCqFree(uint32_t devId, struct halSqCqFreeInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo *info) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqMemGet(uint32_t devId, struct halSqMemGetInput *in, struct halSqMemGetOutput *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqMsgSend(uint32_t devId, struct halSqMsgInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}


drvError_t halCqReportIrqWait(uint32_t devId, struct halReportInfoInput *in, struct halReportInfoOutput *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCqReportGet(uint32_t devId, struct halReportGetInput *in, struct halReportGetOutput *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halReportRelease(uint32_t devId, struct halReportReleaseInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqTaskSend(uint32_t devId, struct halTaskSendInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCqReportRecv(uint32_t devId, struct halReportRecvInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halSqTaskArgsAsyncCopy(uint32_t devId, struct halSqTaskArgsInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halAsyncDmaWqeCreate(uint32_t devId, struct halAsyncDmaInputPara *in, struct halAsyncDmaOutputPara *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halAsyncDmaWqeDestory(uint32_t devId, struct halAsyncDmaDestoryPara *para)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halAsyncDmaCreate(uint32_t devId, struct halAsyncDmaInputPara *in, struct halAsyncDmaOutputPara *out)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halAsyncDmaDestory(uint32_t devId, struct halAsyncDmaDestoryPara *para)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCtl(int cmd, void *param_value, size_t param_value_size, void *out_value, size_t *out_size_ret)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCdqCreate(unsigned int devId, unsigned int tsId, struct halCdqPara *cdqPara,
                        unsigned int *queId) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCdqDestroy(unsigned int devId, unsigned int tsId, unsigned int queId) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCdqAllocBatch(unsigned int devId, unsigned int tsId, unsigned int queId, unsigned int timeout,
                            unsigned int *batchId) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCdqGetReadyBatch(const char *queName, DVdeviceptr *batchAddr, unsigned int *batchSize) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCdqFreeBatch(const char *queName, DVdeviceptr batchAddr) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCdqSetAbnormal(const char *queName) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halCdqGetInstance(const char *queName, unsigned int *instance) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedDettachDevice(unsigned int devId) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedCreateGrpEx(uint32_t devId, struct esched_grp_para *grpPara, unsigned int *grpId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedQueryInfo(unsigned int devId, ESCHED_QUERY_TYPE type, struct esched_input_info *inPut,
                              struct esched_output_info *outPut)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSetGrpEventQos(unsigned int devId, unsigned int grpId, EVENT_ID eventId,
                                   struct event_sched_grp_qos *qos)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSetEventPriority(unsigned int devId, EVENT_ID eventId, SCHEDULE_PRIORITY priority)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedRegisterFinishFunc(unsigned int grpId, unsigned int event_id,
                                       void (*finishFunc)(unsigned int devId, unsigned int grpId, unsigned int event_id,
                                                          unsigned int subevent_id))
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedDumpEventTrace(unsigned int devId, char *buff, unsigned int buffLen, unsigned int *dataLen)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedTraceRecord(unsigned int devId, const char *recordReason, const char *key)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEvent(unsigned int devId, struct event_summary *event) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventToThread(uint32_t devId, struct event_summary *event)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventBatch(unsigned int devId, SUBMIT_FLAG flag, struct event_summary *events,
                                     unsigned int event_num, unsigned int *succ_event_num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventSync(unsigned int devId, struct event_summary *event, int timeout,
                                    struct event_reply *reply) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedThreadSwapout(unsigned int devId, unsigned int grpId, unsigned int threadId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedThreadGiveup(unsigned int devId, unsigned int grpId, unsigned int threadId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedGetEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, EVENT_ID eventId,
                             struct event_info *event)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedAckEvent(unsigned int devId, EVENT_ID eventId, unsigned int subeventId, char *msg,
                             unsigned int msgLen) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedRegisterAckFunc(unsigned int grpId, EVENT_ID eventId,
                                    void (*ackFunc)(unsigned int devId, unsigned int subevent_id, char *msg,
                                                    unsigned int msgLen))
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedTableAddEntry(unsigned int devId, unsigned int tableId, struct esched_table_key *key,
                                  struct esched_table_entry *entry)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedTableDelEntry(unsigned int devId, unsigned int tableId, struct esched_table_key *key)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedTableQueryEntryStat(unsigned int devId, unsigned int tableId, struct esched_table_key *key,
                                        struct esched_table_key_entry_stat *stat)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueSubEvent(struct QueueSubPara *subPara)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueUnsubEvent(struct QueueUnsubPara *unsubPara)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueCreate(unsigned int devId, const QueueAttr *queAttr, unsigned int *qid) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueGrant(unsigned int devId, int qid, int pid, QueueShareAttr attr) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueAttach(unsigned int devId, unsigned int qid, int timeOut) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueue(unsigned int devId, unsigned int qid, void *mbuf) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueue(unsigned int devId, unsigned int qid, void **mbuf) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueSubscribe(unsigned int devId, unsigned int qid, unsigned int groupId, int type) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueUnsubscribe(unsigned int devId, unsigned int qid) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueSubF2NFEvent(unsigned int devId, unsigned int qid, unsigned int groupId) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueUnsubF2NFEvent(unsigned int devId, unsigned int qid) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueGetQidbyName(unsigned int devId, const char *name, unsigned int *qid) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueCtrlEvent(struct QueueSubscriber *subscriber, QUE_EVENT_CMD cmdType) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueuePeek(unsigned int devId, unsigned int qid, uint64_t *buf_len, int timeout) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueuePeekData(unsigned int devId, unsigned int qid, unsigned int flag, QueuePeekDataType type,
                            void **mbuf)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueueBuff(unsigned int devId, unsigned int qid, struct buff_iovec *vector,
                               int timeout) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueueBuff(unsigned int devId, unsigned int qid, struct buff_iovec *vector,
                               int timeout) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEventProc(unsigned int devId, struct event_info *event) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halDrvEventThreadInit(unsigned int devId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halDrvEventThreadUninit(unsigned int devId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueQuery(unsigned int devId, QueueQueryCmdType cmd, QueueQueryInputPara *inPut,
                         QueueQueryOutputPara *outPut) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueSet(unsigned int devId, QueueSetCmdType cmd, QueueSetInputPara *input)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halBufferModeNotify(PSM_STATUS status, void *rsv) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueModeNotify(PSM_STATUS status, void *rsv) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueExport(unsigned int devId, unsigned int qid, struct shareQueInfo *queInfo) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueUnexport(unsigned int devId, unsigned int qid, struct shareQueInfo *queInfo) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueImport(unsigned int devId, struct shareQueInfo *queInfo, unsigned int *qid) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueUnimport(unsigned int devId, unsigned int qid, struct shareQueInfo *queInfo) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halQueueGetDqsQueInfo(unsigned int devId, unsigned int qid, DqsQueueInfo *info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halNotifyGetInfo(uint32_t devId, uint32_t tsId, uint32_t type, uint32_t *val)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetChipCount(int *chip_count) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetChipList(int chip_list[], int count) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetDeviceCountFromChip(int chip_id, int *device_count) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetDeviceFromChip(int chip_id, int device_list[], int count) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetChipFromDevice(int device_id, int *chip_id) 
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetChipInfo(unsigned int devId, halChipInfo *chipInfo)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}
int32_t halMapErrorCode(drvError_t code)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

drvError_t halGetVdevNum(uint32_t *num_dev)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetDevNumEx(uint32_t hw_type, uint32_t *devNum)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetVdevIDs(uint32_t *devices, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetDevIDsEx(uint32_t hw_type, uint32_t *devices, uint32_t len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halGetFaultEvent(uint32_t devId, struct halEventFilter *filter, struct halFaultEventInfo *eventInfo,
                            uint32_t len, uint32_t *eventCount)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halReadFaultEvent(int32_t devId, int timeout, struct halEventFilter *filter,
                             struct halFaultEventInfo *eventInfo)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halParseSDID(uint32_t sdid, struct halSDIDParseInfo *sdid_parse)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}
int uadk_crypto_init(void **ctx, uadk_crypto_param *param, int enc)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int uadk_crypto_alloc(void *ctx_, uadk_mem_info *mem_info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int uadk_crypto_update(void *ctx_, size_t src_len, size_t *dst_len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

void uadk_crypto_free(void *ctx_)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
}

void uadk_crypto_ctx_deinit(void *ctx)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
}

bool halSupportFeature(uint32_t devId, drvFeature_t type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return true;
}

drvError_t halGetHostID(uint32_t *host_id)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

int halGetUserConfig(unsigned int devId, const char *name, unsigned char *buf, unsigned int *bufSize)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halSetUserConfig(unsigned int devId, const char *name, unsigned char *buf, unsigned int bufSize)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halClearUserConfig(unsigned int devId, const char *name)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halGetDeviceVfMax(unsigned int devId, unsigned int *vf_max_num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halGetDeviceVfList(unsigned int devId, unsigned int *vf_list, unsigned int list_len, unsigned int *vf_num)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halTsDevRecord(unsigned int devId, unsigned int tsId, unsigned int record_type, unsigned int record_Id)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

DVresult halMemInitSvmDevice(int hostpid, unsigned int vfid, unsigned int dev_id)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

DVresult halMemBindSibling(int hostPid, int aicpuPid, unsigned int vfid, unsigned int dev_id, unsigned int flag)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResMap(unsigned int devId, struct res_map_info *res_info, unsigned long *va, unsigned int *len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResUnmap(unsigned int devId, struct res_map_info *res_info)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResRead(unsigned int dev_id, struct res_map_info *res_info, void *data, unsigned int len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halResWrite(unsigned int dev_id, struct res_map_info *res_info, void *data, unsigned int len)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halStreamBackup(uint32_t dev_id, struct stream_backup_info *in)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halStreamRestore(uint32_t dev_id, struct stream_backup_info *in)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

int halBuffAllocAlignEx(uint64_t size, unsigned int align, unsigned long flag, int grp_id, void **buff)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

int halBuffFree(void *buff)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return 0;
}

drvError_t halEschedAttachDevice(unsigned int devId)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubscribeEvent(unsigned int devId, unsigned int grpId,
    unsigned int threadId, unsigned long long eventBitmap)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

drvError_t halEschedWaitEvent(unsigned int devId, unsigned int grpId,
    unsigned int threadId, int timeout, struct event_info *event)
{
    printf("[STUB][%s]stub is empty.\n", __func__);
    return DRV_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif  // __cplusplus