#include <store/hccl_sim_world_pub.h>
#include "hccl_vm_log.h"

using namespace HcclSim;

HcclVmResult InitSimWorld(const TopoMeta* topoMeta) {
    ShmSimWorld* simWorld = SHMManager::FindShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD);
    NpuPos2Index* npuPos2IndexMap = SHMManager::FindShmObject<NpuPos2Index>(SHM_MODULE_NPU_IDX_MAP);
    if (simWorld == nullptr || npuPos2IndexMap == nullptr) {
        HCCL_VM_ERROR("SHM obj not found");
        return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
    }

    HCCL_VM_INFO("开始初始化SimWorld...设备数: {:d}", simWorld->devNum);

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simWorld->mutex);

        uint32_t podId = 0;
        size_t npuIdx = 0;
        for (const auto& superPod : (*topoMeta))
        {
            uint32_t serId = 0;
            for (const auto& server : superPod)
            {
                for (const auto& phyId : server)
                {
                    ShmNpuPos npuPos(podId, serId, phyId);
                    ShmSimNpu* npuObj = &simWorld->npu[npuIdx];
                    // 初始化npu资源
                    npuObj->npuPos = npuPos;
                    for (uint32_t i = 0; i < MAX_NOTIFY_NUM; ++i)
                    {
                        npuObj->notify[i].notifyId = ShmNpuResId(npuPos.field.podId, npuPos.field.serId, npuPos.field.phyId,
                                                                 i);
                        npuObj->notify[i].isUsed = false;
                        npuObj->notify[i].value = false;
                    }
                    for (uint32_t i = 0; i < MAX_STREAM_NUM; ++i)
                    {
                        npuObj->stream[i].streamId = ShmNpuResId(npuPos.field.podId, npuPos.field.serId, npuPos.field.phyId,
                                                                 i);
                        npuObj->stream[i].isUsed = false;
                    }
                    for (uint32_t i = 0; i < MAX_MEM_BLOCK_NUM; ++i)
                    {
                        npuObj->memory[i].addr = nullptr;
                        npuObj->memory[i].size = 0;
                        npuObj->memory[i].bufferType = 0;
                        npuObj->memory[i].mockAddr = 0;
                    }
                    npuPos2IndexMap->insert(std::pair<uint32_t, size_t>(npuPos.value, npuIdx));
                    npuIdx++;
                }
                serId++;
            }
            podId++;
        }
    }

    HCCL_VM_INFO("SimWorld初始化完成");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetNpuNum(uint32_t* npuNum) {
    if (npuNum == nullptr) {
        HCCL_VM_ERROR("npuNum is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    SHMManager::InitShm(false);
    ShmSimWorld* simWorld = SHMManager::FindShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD);
    if (simWorld == nullptr) {
        HCCL_VM_ERROR("SHM obj not found");
        return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simWorld->mutex);
        *npuNum = simWorld->devNum;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetNpuByNpuPos(const ShmNpuPos& npuPos, ShmSimNpu** simNpu) {
    if (simNpu == nullptr) {
        HCCL_VM_ERROR("simNpu is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);
    ShmSimWorld* simWorld = SHMManager::FindShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD);
    NpuPos2Index* npuPos2IndexMap = SHMManager::FindShmObject<NpuPos2Index>(SHM_MODULE_NPU_IDX_MAP);
    if (simWorld == nullptr || npuPos2IndexMap == nullptr) {
        HCCL_VM_ERROR("SHM obj not found");
        return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simWorld->mutex);
        const auto it = npuPos2IndexMap->find(npuPos.value);
        if (it == npuPos2IndexMap->end()) {
            HCCL_VM_ERROR("npuPos invalid {}", npuPos.ToString());
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        *simNpu = &(simWorld->npu[it->second]);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclSim::HcclVmResult GetNpuByIndex(uint32_t npuIndex, ShmSimNpu** simNpu)
{
    if (simNpu == nullptr) {
        HCCL_VM_ERROR("simNpu is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    SHMManager::InitShm(false);
    ShmSimWorld* simWorld = SHMManager::FindShmObject<ShmSimWorld>(SHM_MODULE_SIM_WORLD);
    if (simWorld == nullptr) {
        HCCL_VM_ERROR("SHM obj not found");
        return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simWorld->mutex);
        if (npuIndex >= simWorld->devNum) {
            HCCL_VM_ERROR("npuIndex invalid {:d}", npuIndex);
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        *simNpu = &(simWorld->npu[npuIndex]);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult SetCommDomain(uint32_t rankSize, uint32_t rankId, const ShmNpuPos& npuPos) {
    if (rankSize > MAX_DEV_NUM) {
        HCCL_VM_ERROR("rankSize invalid");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    SHMManager::InitShm(false);
    ShmCommDomain* commDomain = SHMManager::FindShmObject<ShmCommDomain>(SHM_MODULE_COMM_DOMAIN);
    if (commDomain == nullptr) {
        HCCL_VM_ERROR("SHM obj not found");
        return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(commDomain->mutex);
        commDomain->rankNum = rankSize;
        commDomain->rankId2NpuPos[rankId] = npuPos;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetNpuPosByRankId(const uint32_t rankId, ShmNpuPos* npuPos) {
    SHMManager::InitShm(false);
    ShmCommDomain* commDomain = SHMManager::FindShmObject<ShmCommDomain>(SHM_MODULE_COMM_DOMAIN);
    if (commDomain == nullptr) {
        HCCL_VM_ERROR("SHM obj not found");
        return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(commDomain->mutex);
        if (rankId >= commDomain->rankNum) {
            HCCL_VM_ERROR("rankId invalid");
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        *npuPos = commDomain->rankId2NpuPos[rankId];
    }

    HCCL_VM_INFO("rankId : {:d} 获取ShmNpuPos : {}", rankId, npuPos->ToString());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetNpuByRankId(uint32_t rankId, ShmSimNpu** simNpu) {
    if (simNpu == nullptr) {
        HCCL_VM_ERROR("simNpu is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    ShmNpuPos npuPos{};
    auto ret = GetNpuPosByRankId(rankId, &npuPos);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS || !npuPos.IsValid()) {
        HCCL_VM_ERROR("GetNpuPosByRankId fail");
        return ret;
    }

    ret = GetNpuByNpuPos(npuPos, simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("GetNpuByNpuPos fail");
        return ret;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclSim::HcclVmResult GetRankIdByNpuPos(const ShmNpuPos& npuPos, uint32_t* rankId)
{
    if (rankId == nullptr) {
        HCCL_VM_ERROR("rankId is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    SHMManager::InitShm(false);
    ShmCommDomain* commDomain = SHMManager::FindShmObject<ShmCommDomain>(SHM_MODULE_COMM_DOMAIN);
    if (commDomain == nullptr) {
        HCCL_VM_ERROR("SHM obj not found");
        return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
    }

    ipc::scoped_lock<ipc::interprocess_mutex> lock(commDomain->mutex);
    for (uint32_t i = 0; i < commDomain->rankNum; ++i) {
        if (commDomain->rankId2NpuPos[i] == npuPos) {
            *rankId = i;
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
    HCCL_VM_ERROR("NpuPos {} not found in CommDomain", npuPos.ToString());
    return HcclVmResult::HCCL_SIM_SHM_OBJ_NOT_FOUND;
}

HcclVmResult AllocNpuMemoryGetIdx(const uint32_t rankId, const uint64_t size, size_t* memBlockIdx) {
    SHMManager::InitShm(false);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByRankId(rankId, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory GetNpuByRankId fail");
        return ret;
    }
    if (simNpu->memCount >= MAX_MEM_BLOCK_NUM) {
        HCCL_VM_ERROR("AllocNpuMemory : npu memblock run out");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    void* memProxy;
    ret = AllocRankMemForProxy(size, &memProxy);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory failed");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        ShmNpuMemBlock& memBlock = simNpu->memory[simNpu->memCount];
        memBlock.addr = memProxy;
        memBlock.size = size;
        *memBlockIdx = simNpu->memCount;
        simNpu->memCount++;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AllocNpuMemory(const uint32_t rankId, const uint64_t size, void** addr) {
    if (addr == nullptr) {
        HCCL_VM_ERROR("addr is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByRankId(rankId, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory GetNpuByRankId fail");
        return ret;
    }
    if (simNpu->memCount >= MAX_MEM_BLOCK_NUM) {
        HCCL_VM_ERROR("AllocNpuMemory : npu memblock run out");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    void* memProxy;
    ret = AllocRankMemForProxy(size, &memProxy);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory failed");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        ShmNpuMemBlock& memBlock = simNpu->memory[simNpu->memCount];
        memBlock.addr = memProxy;
        memBlock.size = size;
        simNpu->memCount++;
        *addr = memBlock.addr.get();
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AllocNpuMemory(const ShmNpuPos& npuPos, const uint64_t size, void** addr) {
    if (addr == nullptr) {
        HCCL_VM_ERROR("addr is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuPos, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory GetNpuByNpuPos fail");
        return ret;
    }
    if (simNpu->memCount >= MAX_MEM_BLOCK_NUM) {
        HCCL_VM_ERROR("AllocNpuMemory : npu memblock run out");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    void* memProxy;
    ret = AllocRankMemForProxy(size, &memProxy);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory failed");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        ShmNpuMemBlock& memBlock = simNpu->memory[simNpu->memCount];
        memBlock.addr = memProxy;
        memBlock.size = size;
        simNpu->memCount++;
        *addr = memBlock.addr.get();
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MockAllocNpuMemory(const uint32_t rankId, const uint64_t size, void** addr) {
    if (addr == nullptr) {
        HCCL_VM_ERROR("addr is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);
    MockProxyBuffer* mockBuffer = SHMManager::FindShmObject<MockProxyBuffer>(SHM_MODULE_MOCK_PROXY_BUFFER);
    if (mockBuffer->nextAddr >= MAX_MOCK_MEM || mockBuffer->mockMemCnt > MAX_MOCK_MEM_BLOCK_NUM) {
        HCCL_VM_ERROR("MockAllocNpuMemory MockProxyBuffer run out");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByRankId(rankId, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("MockAllocNpuMemory GetNpuByRankId fail");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(mockBuffer->mutex);
        mockBuffer->mockMem[mockBuffer->mockMemCnt].npuPos = simNpu->npuPos.value;
        mockBuffer->mockMem[mockBuffer->mockMemCnt].addr = mockBuffer->nextAddr;
        mockBuffer->mockMem[mockBuffer->mockMemCnt].size = size;
        mockBuffer->mockMemCnt++;
        *addr = reinterpret_cast<void*>(mockBuffer->nextAddr);
        size_t fixSize = ((size + MOCK_MEM_GRAN - 1) / MOCK_MEM_GRAN) * MOCK_MEM_GRAN; // 对齐粒度2MB
        mockBuffer->nextAddr += fixSize;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MockAllocNpuMemory(const ShmNpuPos& npuPos, const uint64_t size, void** addr) {
    if (addr == nullptr) {
        HCCL_VM_ERROR("addr is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);
    MockProxyBuffer* mockBuffer = SHMManager::FindShmObject<MockProxyBuffer>(SHM_MODULE_MOCK_PROXY_BUFFER);
    if (mockBuffer->nextAddr >= MAX_MOCK_MEM || mockBuffer->mockMemCnt > MAX_MOCK_MEM_BLOCK_NUM) {
        HCCL_VM_ERROR("MockAllocNpuMemory MockProxyBuffer run out");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(mockBuffer->mutex);
        mockBuffer->mockMem[mockBuffer->mockMemCnt].npuPos = npuPos.value;
        mockBuffer->mockMem[mockBuffer->mockMemCnt].addr = mockBuffer->nextAddr;
        mockBuffer->mockMem[mockBuffer->mockMemCnt].size = size;
        mockBuffer->mockMemCnt++;
        *addr = reinterpret_cast<void*>(mockBuffer->nextAddr);
        size_t fixSize = ((size + MOCK_MEM_GRAN - 1) / MOCK_MEM_GRAN) * MOCK_MEM_GRAN; // 对齐粒度2MB
        mockBuffer->nextAddr += fixSize;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult RegisterNpuMemory(const uint32_t rankId, const void* addr, const uint64_t size, const uint8_t bufferType) {
    HCCL_VM_DEBUG("Register {:p} - Size {:d} - Rank {:d} - Type {:d}", addr, size, rankId, static_cast<int>(bufferType));
    SHMManager::InitShm(false);

    if (bufferType == BufferType::CCL) {
        ShmSimNpu* simNpu = nullptr;
        auto ret = GetNpuByRankId(rankId, &simNpu);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("AllocNpuMemory GetNpuByRankId fail");
            return ret;
        }
        if (simNpu->memCount >= MAX_MEM_BLOCK_NUM) {
            HCCL_VM_ERROR("AllocNpuMemory : npu memblock run out");
            return HcclVmResult::HCCL_SIM_SHM_FAIL;
        }

        {
            ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
            ShmNpuMemBlock& memBlock = simNpu->memory[simNpu->memCount];
            memBlock.mockAddr = reinterpret_cast<uintptr_t>(addr);
            memBlock.size = size;
            memBlock.bufferType = bufferType;
            simNpu->memCount++;
        }
    } else {
        MockProxyBuffer* mockBuffer = SHMManager::FindShmObject<MockProxyBuffer>(SHM_MODULE_MOCK_PROXY_BUFFER);
        if (mockBuffer == nullptr) {
            HCCL_VM_ERROR("RegisterNpuMemory mockBuffer is nullptr");
            return HcclVmResult::HCCL_SIM_SHM_FAIL;
        }
        bool flag = false;
        for (int32_t i = 0; i < mockBuffer->mockMemCnt; i++) {
            if (reinterpret_cast<uint64_t>(addr) >= mockBuffer->mockMem[i].addr
                && reinterpret_cast<uint64_t>(addr) + size <= mockBuffer->mockMem[i].addr + mockBuffer->mockMem[i].size
            ) {
                mockBuffer->mockMem[i].bufferType = bufferType;
                flag = true;
                break;
            }
        }
        if (!flag) {
            HCCL_VM_ERROR("RegisterNpuMemory addr is invalid");
            return HcclVmResult::HCCL_SIM_SHM_FAIL;
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AllocStream(const uint32_t rankId, void** stream) {
    if (stream == nullptr) {
        HCCL_VM_ERROR("stream is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByRankId(rankId, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory GetNpuByRankId fail, rankId: {:d}", rankId);
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        for (int i = 0; i < MAX_STREAM_NUM; ++i) {
            if (simNpu->stream[i].isUsed) {
                continue;
            }
            simNpu->stream[i].isUsed = true;
            *stream = &(simNpu->stream[i]);
            HCCL_VM_INFO("AllocStream : {:p}", *stream);
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
    HCCL_VM_ERROR("AllocStream no Stream left");
    return HcclVmResult::HCCL_SIM_SHM_FAIL;
}

HcclVmResult AllocStream(const ShmNpuPos& npuPos, void** stream) {
    if (stream == nullptr) {
        HCCL_VM_ERROR("stream is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuPos, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocStream GetNpuByNpuPos fail, npuPos : {}", npuPos.ToString());
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        for (int i = 0; i < MAX_STREAM_NUM; ++i) {
            if (simNpu->stream[i].isUsed) {
                continue;
            }
            simNpu->stream[i].isUsed = true;
            *stream = &(simNpu->stream[i]);
            HCCL_VM_INFO("AllocStream : {:p}", *stream);
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
    HCCL_VM_ERROR("AllocStream no Stream left");
    return HcclVmResult::HCCL_SIM_SHM_FAIL;
}

HcclVmResult AllocMainStream(const uint32_t rankId, void** stream) {
    // todo
    if (stream == nullptr) {
        HCCL_VM_ERROR("stream is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    try {
        SHMManager::InitShm(false);
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("AllocMainStream exception: {}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    } catch (...) {
        HCCL_VM_ERROR("AllocMainStream unknow exception");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    ShmNpuPos npuPos;
    ShmSimNpu* simNpu = nullptr;

    auto ret = GetNpuPosByRankId(rankId, &npuPos);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocMainStream : GetNpuPosByRankId fail");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    ret = GetNpuByNpuPos(npuPos, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocMainStream : GetNpuByNpuPos fail");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        if (!simNpu->stream[0].isUsed) {
            *stream = &(simNpu->stream[0]);
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
    HCCL_VM_ERROR("AllocMainStream mainStream is used");
    return HcclVmResult::HCCL_SIM_SHM_FAIL;
}

HcclVmResult AllocSlaveStream(const uint32_t rankId, void** stream) {
    // todo
    if (stream == nullptr) {
        HCCL_VM_ERROR("stream is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    try {
        SHMManager::InitShm(false);
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("AllocSlaveStream exception: {}", e.what());
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    } catch (...) {
        HCCL_VM_ERROR("AllocSlaveStream unknow exception");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    ShmNpuPos npuPos;
    ShmSimNpu* simNpu = nullptr;

    auto ret = GetNpuPosByRankId(rankId, &npuPos);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocSlaveStream : GetNpuPosByRankId fail");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    ret = GetNpuByNpuPos(npuPos, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocSlaveStream : GetNpuByNpuPos fail");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        for (int i = 1; i < MAX_STREAM_NUM; ++i) {
            if (simNpu->stream[i].isUsed) {
                continue;
            }
            simNpu->stream[i].isUsed = true;
            *stream = &(simNpu->stream[i]);
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
    HCCL_VM_ERROR("AllocSlaveStream no SlaveStream left");
    return HcclVmResult::HCCL_SIM_SHM_FAIL;
}

HcclVmResult ReleaseStream(void* stream) {
    ShmSimStream* streamObj = static_cast<ShmSimStream*>(stream);
    ShmNpuResId npuResId = streamObj->streamId;

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuResId.GetNpuPos(), &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("ReleaseStream GetNpuByNpuPos fail");
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        simNpu->stream[npuResId.field.resId].isUsed = false;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ReleaseStream(const uint64_t streamId) {
    ShmNpuResId npuResId(streamId);
    
    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuResId.GetNpuPos(), &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("ReleaseStream GetNpuByNpuPos fail");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        simNpu->stream[npuResId.field.resId].isUsed = false;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AllocNotify(const uint32_t rankId, void** notify) {
    if (notify == nullptr) {
        HCCL_VM_ERROR("notify is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByRankId(rankId, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNpuMemory GetNpuByRankId fail");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        for (int i = 0; i < MAX_NOTIFY_NUM; ++i) {
            if (simNpu->notify[i].isUsed) {
                continue;
            }
            simNpu->notify[i].isUsed = true;
            simNpu->notify[i].value = false;
            *notify = &(simNpu->notify[i]);
            HCCL_VM_INFO("AllocNotify : {:p}", *notify);
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
    HCCL_VM_ERROR("AllocNotify no Notify left");
    return HcclVmResult::HCCL_SIM_SHM_FAIL;
}

HcclVmResult AllocNotify(const ShmNpuPos& npuPos, void** notify) {
    if (notify == nullptr) {
        HCCL_VM_ERROR("notify is nullptr");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    
    SHMManager::InitShm(false);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuPos, &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("AllocNotify GetNpuByNpuPos fail, npuPos : {}", npuPos.ToString());
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        for (int i = 0; i < MAX_NOTIFY_NUM; ++i) {
            if (simNpu->notify[i].isUsed) {
                continue;
            }
            simNpu->notify[i].isUsed = true;
            simNpu->notify[i].value = false;
            *notify = &(simNpu->notify[i]);
            HCCL_VM_INFO("AllocNotify : {:p}", *notify);
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
    HCCL_VM_ERROR("AllocNotify no Notify left");
    return HcclVmResult::HCCL_SIM_SHM_FAIL;
}

HcclVmResult ReleaseNotify(void* notify) {
    ShmSimNotify* notifyObj = static_cast<ShmSimNotify*>(notify);
    ShmNpuResId npuResId = notifyObj->notifyId;
    
    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuResId.GetNpuPos(), &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("ReleaseNotify : GetNpuByNpuPos fail");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        simNpu->notify[npuResId.field.resId].isUsed = false;
        simNpu->notify[npuResId.field.resId].value = false;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ReleaseNotify(const uint64_t notifyId) {
    ShmNpuResId npuResId(notifyId);
    
    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuResId.GetNpuPos(), &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("ReleaseNotify : GetNpuByNpuPos fail");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        simNpu->notify[npuResId.field.resId].isUsed = false;
        simNpu->notify[npuResId.field.resId].value = false;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetNotifyValue(const uint64_t notifyId, bool* value) {
    ShmNpuResId npuResId(notifyId);
    
    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuResId.GetNpuPos(), &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("ReleaseNotify : GetNpuByNpuPos fail");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        *value = simNpu->notify[npuResId.field.resId].value;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult SetNotifyValue(const uint64_t notifyId, bool value) {
    ShmNpuResId npuResId(notifyId);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuResId.GetNpuPos(), &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("ReleaseNotify : GetNpuByNpuPos fail");
        return ret;
    }

    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        simNpu->notify[npuResId.field.resId].value = value;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult WaitNotifyValue(const uint64_t notifyId, bool* result) {
    ShmNpuResId npuResId(notifyId);

    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByNpuPos(npuResId.GetNpuPos(), &simNpu);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("ReleaseNotify : GetNpuByNpuPos fail");
        return ret;
    }
    {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(simNpu->mutex);
        if (simNpu->notify[npuResId.field.resId].value == false) {
            *result = false;
            return HcclVmResult::HCCL_SIM_SUCCESS;
        } else {
            simNpu->notify[npuResId.field.resId].value = false;
            *result = true;
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
    }
}