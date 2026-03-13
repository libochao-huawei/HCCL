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
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
// #include "hccl_vm.h"
#include "task_ventilator.h"
#include "sim_models.h"
#include "sim_runner_ops.h"
#include <securec.h>
#include "rt_external_mem.h"
#include "hccl_vm_log.h"
#include "hccl_proxy_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void* GetRealPtrByAddr(const void *devPtr)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)devPtr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([startPtr](const sim::VirtualMemBlock &virMem) {
        return (virMem.start_ptr <= startPtr) && (virMem.start_ptr + virMem.size > startPtr);
    });
    if (!virMemRes.second) {
        return nullptr;
    }

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(virMemRes.first.phy_mem_id);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[GetRealPtrByAddr] cannot find phy Mem id:{:d}", virMemRes.first.phy_mem_id);
        return nullptr;
    }

    uint64_t ptrDiff = startPtr - virMemRes.first.start_ptr;

    void* realPtr = nullptr;
    GetPhyPtrFromOffsetPtr(&realPtr, phyMemRes->addr);
    if (realPtr == nullptr) {
        return nullptr;
    }

    return (void *)((char*)realPtr + ptrDiff);
}

bool GetOffsetAndCtxIdFromPtr(void *devPtr, uint64_t* offsetPtr, uint64_t* ctxId)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)devPtr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([startPtr](const sim::VirtualMemBlock &virMem) {
        return (virMem.start_ptr <= startPtr) && (virMem.start_ptr + virMem.size > startPtr);
    });
    if (!virMemRes.second) {
        return false;
    }

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(virMemRes.first.phy_mem_id);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[GetRealPtrByAddr] cannot find phy Mem id:{:d}", virMemRes.first.phy_mem_id);
        return false;
    }

    uint64_t ptrDiff = startPtr - virMemRes.first.start_ptr;
    *offsetPtr =  phyMemRes->addr + ptrDiff;
    *ctxId = virMemRes.first.ctx_id;
    return true;
}
////////////////////////////////////////////////////

aclError aclrtMallocHost(void **hostPtr, size_t size)
{
    uint64_t offset_ptr = 0;
    void* realPtr = nullptr;
    HcclSim::HcclVmResult ret = AllocatePhy(&realPtr, &offset_ptr, size);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[aclstub][aclrtMallocHost] MockAllocNpuMemory Failed.");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // runner
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    uint64_t deviceIdx = 0xFFFFFFF;
    if (!currCtx.has_value()) {
        HCCL_VM_WARN("[aclrtMallocHost] can not find current context:{:d}", runner.current_ctx_id);
        deviceIdx = currCtx->device_id;
 
    }

    sim::PhyMemBlock phyMem{};
    phyMem.device_id = deviceIdx;
    phyMem.addr = offset_ptr;
    phyMem.size = size;
    phyMem.ref_count = 1;
    auto phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = (uint64_t)(uintptr_t)realPtr;
    virMem.size = size;
    virMem.ctx_id   = runner.current_ctx_id;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = runner.pid;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_HOST;
    virMem.policy = 0;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    *hostPtr = realPtr;

    HCCL_VM_INFO("aclrtMallocHost...., addr:{:p}, offsetPtr:0x{:x}, size:{:d}" , realPtr, offset_ptr, size);
    return ACL_SUCCESS;
}

aclError aclrtMallocHostWithCfg(void **ptr, uint64_t size, aclrtMallocConfig *cfg)
{
    return aclrtMallocHost(ptr, size);
}

aclError aclrtFreeHost(void *hostPtr)
{
    uint64_t start_ptr = (uint64_t)(uintptr_t)hostPtr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [start_ptr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  start_ptr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_HOST;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[ERROR][aclrtFreeHost] can not find this buff offset ptr:0x{:x}", start_ptr);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(virMemRes.first.phy_mem_id);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[ERROR][FreeStub] cannot find phy Mem id:{:d}", virMemRes.first.phy_mem_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    HCCL_VM_INFO("[aclrtFreeHost] hostPtr:{:p} offsetPtr:0x{:x}", hostPtr, phyMemRes->addr);

    RunnerDB::Delete<sim::VirtualMemBlock>(virMemRes.first.id);
    RunnerDB::Delete<sim::PhyMemBlock>(phyMemRes->id);

    DeallocatePhy(hostPtr, phyMemRes->addr, phyMemRes->size);
    return ACL_SUCCESS;
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    uint64_t offset_ptr = 0;
    void* realPtr = nullptr;
    HcclSim::HcclVmResult ret = AllocatePhy(&realPtr, &offset_ptr, size);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[aclstub][aclrtMallocHost] MockAllocNpuMemory Failed.");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // runner
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        DeallocatePhy(realPtr, offset_ptr, size);
        HCCL_VM_ERROR("[ERROR][aclrtMallocHost] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::PhyMemBlock phyMem{};
    phyMem.device_id = currCtx->device_id;
    phyMem.addr = offset_ptr;
    phyMem.size = size;
    phyMem.ref_count = 1;
    auto phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    void* virPtr = nullptr;
    AllocateVir(&virPtr, size);
    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = (uint64_t)(uintptr_t)virPtr;
    virMem.size = size;
    virMem.ctx_id   = runner.current_ctx_id;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = runner.pid;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.policy = policy;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    *devPtr = virPtr;
    HCCL_VM_INFO("rtMalloc...., addr:{:p}", virPtr);
    return ACL_SUCCESS;
}

aclError aclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    size_t realSize = (size % 32 + 1) * 32;
    return aclrtMalloc(devPtr, realSize, policy);
}

aclError aclrtMallocCached(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtMemFlush(void *devPtr, size_t size)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtMemInvalidate(void *devPtr, size_t size)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtMallocWithCfg(void **devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig *cfg)
{
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtMallocForTaskScheduler(void **devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig *cfg)
{
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtFree(void* devPtr)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)devPtr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [startPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[ERROR][aclrtFree] can not find this buff offset ptr:0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(virMemRes.first.phy_mem_id);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[ERROR][FreeStub] cannot find phy Mem id:{:d}", virMemRes.first.phy_mem_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    RunnerDB::Delete<sim::VirtualMemBlock>(virMemRes.first.id);
    RunnerDB::Delete<sim::PhyMemBlock>(phyMemRes->id);
    DeallocatePhy(devPtr, phyMemRes->addr, phyMemRes->size);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    void* realPtr = GetRealPtrByAddr(devPtr);
    if (realPtr == nullptr) {
        HCCL_VM_ERROR("[ERROR][aclrtMemset] cannot find vir Mem devPtr:{:p}", devPtr);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    memset_s(realPtr, maxCount, value, count);
    HCCL_VM_INFO("[aclstub][aclrtMemset] ptr:{:p}, maxCount: {:d}, value: {:d}, count: {:d}", devPtr, maxCount, value, count);
    return ACL_SUCCESS;
}

aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream)
{
    // TODO 当前先不生成任务，后续有需要再根据实际情况生成任务
    return aclrtMemset(devPtr, maxCount, value, count);
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    void* realSrc = GetRealPtrByAddr(src);
    if (realSrc == nullptr) {
        HCCL_VM_WARN("[aclrtMemcpy]stub cannot find vir Mem src:{:p} this is malloc addr", src);
        realSrc = const_cast<void*>(src);
    }

    void* realDst = GetRealPtrByAddr(dst);
    if (realDst == nullptr) {
        HCCL_VM_WARN("[aclrtMemcpy]stub cannot find vir Mem dst:{:p} this is malloc addr", dst);
        realDst = dst;
    }
    memcpy(realDst, realSrc, count);
    HCCL_VM_INFO("[aclrtMemcpy]stub src:{:p} to dst:{:p}, size:{:d}", realSrc, realDst, count);
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    if (kind == ACL_MEMCPY_HOST_TO_HOST) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t srcCtxId = 0;
    uint64_t dstCtxId = 0;
    if(!GetOffsetAndCtxIdFromPtr(const_cast<void*>(src), &srcOffset, &srcCtxId) || !GetOffsetAndCtxIdFromPtr(dst, &dstOffset, &dstCtxId)) {
        HCCL_VM_ERROR("[ERROR][aclrtMemcpyAsync] cannot find vir Mem dst:{:p}, src:{:p}", dst, src);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint64_t streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);
    HcclTaskMetaData taskMeta;
    taskMeta.rankId = curRank;
    taskMeta.streamId = streamId;
    taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta.taskData.transMem.srcOffset = (uint64_t)(uintptr_t)src;
    taskMeta.taskData.transMem.dstOffset = (uint64_t)(uintptr_t)dst;
    taskMeta.taskData.transMem.len = count;
    taskMeta.taskData.transMem.srcRankId = sim::GetRankIdByCtxId(srcCtxId);
    taskMeta.taskData.transMem.dstRankId = sim::GetRankIdByCtxId(dstCtxId);

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMeta, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[ERROR] [{:s}] InsertTaskToCollection fail", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    sim::Task task{};
    task.stream_id  = streamId;
    task.cid        = taskCid.value;
    task.type       = (uint8_t)HccLTaskMetaType::MEM_CPY;

    auto taskId = RunnerDB::Add<sim::Task>(task);

    // TODO
    // 下发cid
    //TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);
    // 记录状态
    //TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithCondition(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    return aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
}

aclError aclrtMemcpyBatch(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes, size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, size_t *failIndex)
{
    for (size_t i = 0; i < numBatches; i++) {
        aclrtMemcpy(dsts[i], destMaxs[i], srcs[i], sizes[i], ACL_MEMCPY_DEFAULT);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpyBatchAsync(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes, size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, size_t *failIndex, aclrtStream stream)
{
    for (size_t i = 0; i < numBatches; i++) {
        aclrtMemcpyAsync(dsts[i], destMaxs[i], srcs[i], sizes[i], ACL_MEMCPY_DEFAULT, stream);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind)
{
    if (kind != ACL_MEMCPY_HOST_TO_DEVICE || kind != ACL_MEMCPY_DEVICE_TO_HOST) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    // TODO
    return ACL_SUCCESS;
}

aclError aclrtMemcpy2dAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind, aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetMemcpyDescSize(aclrtMemcpyKind kind, size_t *descSize)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtSetMemcpyDesc(void *desc, aclrtMemcpyKind kind, void *srcAddr, void *dstAddr, size_t count, void *config)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithDesc(void *desc, aclrtMemcpyKind kind, aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags)
{
    uint64_t offset_ptr = 0;
    void* realPtr = nullptr;
    HcclSim::HcclVmResult ret = AllocatePhy(&realPtr, &offset_ptr, size);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[aclstub][MallocStub] MockAllocNpuMemory Failed.");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // runner
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        DeallocatePhy(realPtr, offset_ptr, size);
        HCCL_VM_ERROR("[ERROR][aclrtMallocPhysical] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::PhyMemBlock phyMem{};
    phyMem.device_id = currCtx->device_id;
    phyMem.addr = offset_ptr;
    phyMem.size = size;
    phyMem.ref_count = 1;
    auto phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);
    *handle = (aclrtDrvMemHandle)phyMemId;
    return ACL_SUCCESS;
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
{
    uint64_t phyMemId = (uint64_t)(uintptr_t)handle;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[ERROR][FreeStub] cannot find phy Mem offset:{:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    RunnerDB::Delete<sim::PhyMemBlock>(phyMemRes->id);
    DeallocatePhy(nullptr, phyMemRes->addr, phyMemRes->size);
    return ACL_SUCCESS;
}

aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags)
{
    // runner
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("[ERROR][aclrtReserveMemAddress] can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    void* startPtr = nullptr;
    AllocateVir(&startPtr, size);
    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = (uint64_t)(uintptr_t)startPtr;
    virMem.size = size;
    virMem.ctx_id   = runner.current_ctx_id;
    virMem.owner_pid = runner.pid;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);
    *virPtr = startPtr;
    return ACL_SUCCESS;
}

aclError aclrtReleaseMemAddress(void *virPtr)
{
    uint64_t starPtr = (uint64_t)(uintptr_t)virPtr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [starPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr == starPtr;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclstub][aclrtReleaseMemAddress] can not find the vir ptr");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    RunnerDB::Delete<sim::VirtualMemBlock>(virMemRes.first.id);
    return ACL_SUCCESS;
}

aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)virPtr;
    uint64_t phyMemId = (uint64_t)(uintptr_t)handle;

    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [startPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclrtMapMem] can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[aclrtMapMem] cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto runner = sim::GetCurrRunnerTls();
    sim::FdMemRecord memRecord{};
    memRecord.vir_mem_id = virMemRes.first.id;
    memRecord.phy_mem_id = phyMemId;
    memRecord.create_pid = runner.pid;
    RunnerDB::Add<sim::FdMemRecord>(memRecord);

    RunnerDB::Update<sim::VirtualMemBlock>(virMemRes.first.id, [phyMemId](sim::VirtualMemBlock &virMem) {
        virMem.phy_mem_id = phyMemId;
    });

    return ACL_SUCCESS;
}

aclError aclrtUnmapMem(void *virPtr)
{
    auto runner = sim::GetCurrRunnerTls();
    auto pid = runner.pid;
    uint64_t startPtr = (uint64_t)(uintptr_t)virPtr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [startPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclrtUnmapMem] can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    RunnerDB::Update<sim::VirtualMemBlock>(virMemRes.first.id, [](sim::VirtualMemBlock &virMem) {
        virMem.phy_mem_id = 0;
    });

    auto virMemId = virMemRes.first.id;
    auto phyMemId = virMemRes.first.phy_mem_id;
    auto recordRes =
        RunnerDB::GetOneByPred<sim::FdMemRecord>([virMemId, phyMemId, pid](const sim::FdMemRecord &record) {
            return record.vir_mem_id == virMemId && record.phy_mem_id == phyMemId && record.create_pid == pid;
        });
    if (!recordRes.second) {
        HCCL_VM_ERROR("[aclrtUnmapMem] can not find this buff virptr: 0x{:x} phy: {:d} pid: {:d}", virMemId, phyMemId, pid);
        return ACL_ERROR_INVALID_PARAM;
    }

    RunnerDB::Delete<sim::FdMemRecord>(recordRes.first.id);
    return ACL_SUCCESS;
}

aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType handleType, uint64_t flags, uint64_t *shareableHandle)
{
    uint64_t phyMemId = (uint64_t)(uintptr_t)handle;

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[aclrtMemExportToShareableHandle] cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *shareableHandle = phyMemRes->id;
    return ACL_SUCCESS;
}

aclError aclrtDeviceGetBareTgid(int32_t *pid)
{
    auto runner = sim::GetCurrRunnerTls();
    *pid = runner.pid;
    return ACL_SUCCESS;
}

aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t *pid, size_t pidNum)
{
    auto runner = sim::GetCurrRunnerTls();
    for (size_t i = 0; i < pidNum; i++) {
        sim::FdMemWhiteList tmp{};
        tmp.name_or_key = shareableHandle;
        tmp.pid = pid[i];
        tmp.create_pid = runner.pid;
        RunnerDB::Add<sim::FdMemWhiteList>(tmp);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle *handle)
{
    uint64_t phyMemId = (uint64_t)shareableHandle;

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[aclrtMemImportFromShareableHandle] cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *handle = (aclrtDrvMemHandle)phyMemId;
    return ACL_SUCCESS;
}

aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop, aclrtMemGranularityOptions option, size_t *granularity)
{
    // 1字节对齐
    *granularity = 1;
    return ACL_SUCCESS;
}

aclError aclrtCmoAsync(void *src, size_t size, aclrtCmoType cmoType, aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtCmoAsyncWithBarrier(void *src, size_t size, aclrtCmoType cmoType, uint32_t barrierId, aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtCmoWaitBarrier(aclrtBarrierTaskInfo *taskInfo, aclrtStream stream, uint32_t flag)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtPointerGetAttributes(const void *ptr, aclrtPtrAttributes *attributes)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)ptr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [startPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_HOST;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclrtPointerGetAttributes] can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[aclrtPointerGetAttributes] cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(phyMemRes->device_id);
    if (!devRes.has_value()) {
        HCCL_VM_ERROR("[aclrtPointerGetAttributes] cannot find phy Mem device: {:d}", phyMemRes->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    attributes->location.id = devRes->logic_id;
    attributes->location.type = (aclrtMemLocationType)(virMemRes.first.src_type);
    return ACL_SUCCESS;
}

aclError aclrtHostRegister(void *ptr, uint64_t size, aclrtHostRegisterType type, void **devPtr)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)ptr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [startPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_HOST;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclrtHostRegister] can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    *devPtr = ptr;
    return ACL_SUCCESS;
}

aclError aclrtHostUnregister(void *ptr)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)ptr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [startPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_HOST;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclrtHostUnregister] can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtValueWrite(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtValueWait(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len, uint64_t flags)
{
    uint64_t startPtr = (uint64_t)(uintptr_t)devPtr;
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [startPtr](const sim::VirtualMemBlock &virMem) { return virMem.start_ptr ==  startPtr;});
    if (!virMemRes.second) {
        HCCL_VM_ERROR("[aclrtIpcMemGetExportKey] can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("[aclrtIpcMemGetExportKey] cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto runner = sim::GetCurrRunnerTls();
    sim::IpcMemRecord memRecord{};
    memRecord.vir_mem_id = virMemRes.first.id;
    memRecord.create_pid = runner.pid;
    auto recordIdx = RunnerDB::Add<sim::IpcMemRecord>(memRecord);

    *(uint64_t *)key = recordIdx;

    return ACL_SUCCESS;
}

aclError aclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num)
{
    auto runner = sim::GetCurrRunnerTls();
    for (int32_t i = 0; i < num; i++) {
        sim::IpcMemWhiteList tmp{};
        tmp.name_or_key = *(uint64_t*)key;
        tmp.pid = pid[i];
        tmp.create_pid = runner.pid;
        RunnerDB::Add<sim::IpcMemWhiteList>(tmp);
    }
    return ACL_SUCCESS;
}

aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flags)
{
    uint64_t ipcRecordIdx = *(const uint64_t *)key;
    auto recordRes = RunnerDB::GetById<sim::IpcMemRecord>(ipcRecordIdx);
    if (!recordRes.has_value()) {
        HCCL_VM_ERROR("[aclrtIpcMemImportByKey] cannot find ipc record: {:d}", ipcRecordIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto virMemRes = RunnerDB::GetById<sim::VirtualMemBlock>(recordRes->vir_mem_id);
    if (!virMemRes.has_value()) {
        HCCL_VM_ERROR("[aclrtIpcMemImportByKey] cannot find vir mem: {:d}", recordRes->vir_mem_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    *devPtr = (void *)virMemRes->start_ptr;

    return ACL_SUCCESS;
}

aclError aclrtIpcMemClose(const char *key)
{
    uint64_t ipcRecordIdx = *(const uint64_t *)key;
    auto recordRes = RunnerDB::GetById<sim::IpcMemRecord>(ipcRecordIdx);
    if (!recordRes.has_value()) {
        HCCL_VM_ERROR("[aclrtIpcMemClose] cannot find ipc record: {:d}", ipcRecordIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto runner = sim::GetCurrRunnerTls();
    if (runner.pid == recordRes->create_pid) {
        RunnerDB::Delete<sim::IpcMemRecord>(recordRes->id);
    }

    return ACL_SUCCESS;
}

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtAllocatorRegister(aclrtStream stream, aclrtAllocatorDesc allocatorDesc)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtAllocatorGetByStream(aclrtStream stream, aclrtAllocatorDesc *allocatorDesc, aclrtAllocator *allocator, aclrtAllocatorAllocFunc *allocFunc, aclrtAllocatorFreeFunc *freeFunc, aclrtAllocatorAllocAdviseFunc *allocAdviseFunc, aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtAllocatorUnregister(aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithOffsetImpl(void **dst, size_t destMax, uint64_t dstDataOffset, const void **src,
    size_t count, size_t srcDataOffset, aclrtMemcpyKind kind, aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtIpcMemSetAttr(const char *key, aclrtIpcMemAttrType type, uint64_t attr)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtIpcMemImportPidInterServer(const char *name, aclrtServerPid *serverPids, size_t num)
{
    const aclrtServerPid &rtServerPid = *serverPids;
    return aclrtIpcMemSetImportPid(name, rtServerPid.pid, rtServerPid.num);
}

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    int ret = aclrtMalloc(devPtr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    HCCL_VM_INFO("zhf-enter into stub rtMalloc.... addr:{:p}", *devPtr);
    return ret;
}

aclError rtFree(void* devPtr)
{
    HCCL_VM_ERROR("[aclstub] enter into rtFree not support");
    return ACL_ERROR_INTERNAL_ERROR;
}

std::map<uint8_t, HcclDataType> rtDataType2CheckerDataType = {
    { aclDataType::ACL_FLOAT, HcclDataType::HCCL_DATA_TYPE_FP32 },
    { aclDataType::ACL_FLOAT16, HcclDataType::HCCL_DATA_TYPE_FP16 },
    { aclDataType::ACL_INT8,  HcclDataType::HCCL_DATA_TYPE_INT8 },
    { aclDataType::ACL_INT32, HcclDataType::HCCL_DATA_TYPE_INT32 },
    { aclDataType::ACL_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8 },
    { aclDataType::ACL_INT16, HcclDataType::HCCL_DATA_TYPE_INT16 },
    { aclDataType::ACL_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16 },
    { aclDataType::ACL_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32 },
    { aclDataType::ACL_INT64,  HcclDataType::HCCL_DATA_TYPE_INT64 },
    { aclDataType::ACL_BF16,  HcclDataType::HCCL_DATA_TYPE_BFP16 }
};

aclError aclrtReduceAsync(void *dst, const void *src, uint64_t count, aclrtReduceKind kind, aclDataType type, aclrtStream stream, void *reserve)
{
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t srcCtxId = 0;
    uint64_t dstCtxId = 0;
    if(!GetOffsetAndCtxIdFromPtr(const_cast<void*>(src), &srcOffset, &srcCtxId) || !GetOffsetAndCtxIdFromPtr(dst, &dstOffset, &dstCtxId)) {
        HCCL_VM_ERROR("[ERROR][aclrtReduceAsync] cannot find vir Mem dst:{:p}, src:{:p}", dst, src);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    auto iter = rtDataType2CheckerDataType.find(type);
    if (iter == rtDataType2CheckerDataType.end()) {
        HCCL_VM_ERROR("[ERROR][aclrtReduceAsync] not support data type: {}", static_cast<uint32_t>(type));
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    auto dataType = iter->second;
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    uint64_t streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.reduce.srcRankId = sim::GetRankIdByCtxId(srcCtxId);
    taskMetaData.taskData.reduce.dstRankId = sim::GetRankIdByCtxId(dstCtxId);
    taskMetaData.taskData.reduce.srcOffset = (uint64_t)(uintptr_t)src;
    taskMetaData.taskData.reduce.dstOffset = (uint64_t)(uintptr_t)dst;
    taskMetaData.taskData.reduce.dataType  = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.dataCount = count / dataSize;
    taskMetaData.taskData.reduce.reduceOp  = static_cast<uint8_t>(kind);

    uint32_t index{0};
    HCCL_VM_DEBUG("[aclstub][aclrtReduceAsync] Get reduce task, streamId={:d}", streamId);
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[{}] InsertTaskToCollection fail", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    sim::Task task{};
    task.stream_id  = streamId;
    task.cid        = taskCid.value;
    task.type       = (uint8_t)HccLTaskMetaType::REDUCE;

    auto taskId = RunnerDB::Add<sim::Task>(task);
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus
