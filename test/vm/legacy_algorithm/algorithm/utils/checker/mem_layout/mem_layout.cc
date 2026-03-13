#include "mem_layout.h"

#include <limits.h>

#include "rank_info_recorder.h"
#include "dtype_common.h"
#include "log.h"

using namespace Hccl;

namespace checker {

MemLayout* MemLayout::Global()
{
    static MemLayout* globalMemSim = new MemLayout;
    return globalMemSim;
}

HcclResult MemLayout::GetSlice(char_t *addr, u64 dataCount, const HcclDataType dataType, DataSlice& dataSlice, RankId* rank)
{
    RankId curRank = GetRankIdByAddr(addr);
    if (rank != nullptr) {
        *rank = curRank;
    }

    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout &singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    for (u32 index = 0; index < singleRankMemLayout.size(); index++) {
        if (addr < singleRankMemLayout[index].startAddr) {
            continue;
        }

        u64 curBlockEndAddr = (u64)(singleRankMemLayout[index].startAddr) + singleRankMemLayout[index].size;
        if (curBlockEndAddr < u64(addr)) {
            continue;
        }

        u64 size = dataCount * SIZE_TABLE[dataType];

        if (singleRankMemLayout[index].bufferType == BufferType::USERBUF_AIV) {
            u32 blockId = GetBlockIdByAddr(addr);
            curBlockEndAddr =
                (u64)allRanksUBMemLayout[curRank][blockId].startAddr + allRanksUBMemLayout[curRank][blockId].size;
            if ((u64)addr + size > curBlockEndAddr) {
                HCCL_ERROR("addr[%p], size[%llu] exceed mem bufferType[%s]: startAddr[%p], endAddr[%p], size[%llu]",
                    addr,
                    size,
                    allRanksUBMemLayout[curRank][blockId].bufferType.Describe().c_str(),
                    allRanksUBMemLayout[curRank][blockId].startAddr,
                    curBlockEndAddr,
                    allRanksUBMemLayout[curRank][blockId].size);
                return HcclResult::HCCL_E_MEMORY;
            }
            dataSlice.SetBufferType(allRanksUBMemLayout[curRank][blockId].bufferType);
            dataSlice.SetOffset(addr - allRanksUBMemLayout[curRank][0].startAddr);
            dataSlice.SetSize(size);
            return HcclResult::HCCL_SUCCESS;
        }

        // 超出这个block的地址范围
        if ((u64)addr + size > curBlockEndAddr) {
            HCCL_ERROR("addr[%p], size[%llu] exceed mem bufferType[%s]: startAddr[%p], endAddr[%p], size[%llu]",
                addr,
                size,
                singleRankMemLayout[index].bufferType.Describe().c_str(),
                singleRankMemLayout[index].startAddr,
                curBlockEndAddr,
                singleRankMemLayout[index].size);
            return HcclResult::HCCL_E_MEMORY;
        }

        dataSlice.SetBufferType(singleRankMemLayout[index].bufferType);
        dataSlice.SetOffset(addr - singleRankMemLayout[index].startAddr);
        dataSlice.SetSize(size);
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_ERROR("fail to get dataSlice, addr= %p", addr);
    return HcclResult::HCCL_E_MEMORY;
}

#ifndef HCCL_ALG_ANALYZER_DAVID
HcclResult MemLayout::GetSlice(const DeviceMem &deviceMem, DataSlice& dataSlice, RankId* rank)
{
    u64 addr = (u64)deviceMem.ptr();
    u64 size = deviceMem.size();

    RankId curRank = GetRankIdByAddr((char*)deviceMem.ptr());
    if (rank != nullptr) {
        *rank = curRank;
    }

    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout& singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    for (u32 index = 0; index < singleRankMemLayout.size(); index++) {
        if (addr < (u64)singleRankMemLayout[index].startAddr) {
            continue;
        }

        u64 curBlockEndAddr = (u64)(singleRankMemLayout[index].startAddr) + singleRankMemLayout[index].size;
        if (curBlockEndAddr < u64(addr)) {
            continue;
        }

        // 超出这个block的地址范围
        if ((u64)addr + size > curBlockEndAddr) {
            HCCL_ERROR("addr[%p], size[%llu] exceed mem bufferType[%s]: startAddr[%p], endAddr[%p], size[%llu]",
                       addr, size, singleRankMemLayout[index].bufferType.Describe().c_str(),
                       singleRankMemLayout[index].startAddr, curBlockEndAddr, singleRankMemLayout[index].size);
            return HcclResult::HCCL_E_MEMORY;
        }

        dataSlice.SetBufferType(singleRankMemLayout[index].bufferType);
        dataSlice.SetOffset(addr - (u64)singleRankMemLayout[index].startAddr);
        dataSlice.SetSize(size);

        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_ERROR("fail to get dataSlice, addr= %p", addr);
    return HcclResult::HCCL_E_MEMORY;
}
#endif

HcclResult MemLayout::GetSlice(char_t *addr, u64 len, DataSlice &dataSlice, RankId *rank)
{
    // HCCL_ERROR("Enter to get dataSlice, addr= %p", addr);
    RankId curRank = GetRankIdByAddr(addr);
    if (rank != nullptr) {
        *rank = curRank;
    }

    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout &singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    for (u32 index = 0; index < singleRankMemLayout.size(); index++) {
        if (addr < singleRankMemLayout[index].startAddr) {
            continue;
        }

        u64 curBlockEndAddr = (u64)(singleRankMemLayout[index].startAddr) + singleRankMemLayout[index].size;
        if (curBlockEndAddr < u64(addr)) {
            continue;
        }

        if (singleRankMemLayout[index].bufferType == BufferType::USERBUF_AIV) {
            u32 blockId = GetBlockIdByAddr(addr);
            curBlockEndAddr =
                (u64)allRanksUBMemLayout[curRank][blockId].startAddr + allRanksUBMemLayout[curRank][blockId].size;
            if ((u64)addr + len > curBlockEndAddr) {
                HCCL_ERROR("addr[%p], size[%llu] exceed mem bufferType[%s]: startAddr[%p], endAddr[%p], size[%llu]",
                    addr,
                    len,
                    allRanksUBMemLayout[curRank][blockId].bufferType.Describe().c_str(),
                    allRanksUBMemLayout[curRank][blockId].startAddr,
                    curBlockEndAddr,
                    allRanksUBMemLayout[curRank][blockId].size);
                return HcclResult::HCCL_E_MEMORY;
            }
            dataSlice.SetBufferType(allRanksUBMemLayout[curRank][blockId].bufferType);
            dataSlice.SetOffset(addr - allRanksUBMemLayout[curRank][0].startAddr);
            dataSlice.SetSize(len);
            return HcclResult::HCCL_SUCCESS;
        }

        // 超出这个block的地址范围
        if ((u64)addr + len > curBlockEndAddr) {
            HCCL_ERROR("addr[%p], len[%llu] exceed mem bufferType[%s]: startAddr[%p], endAddr[%p], size[%llu]",
                addr,
                len,
                singleRankMemLayout[index].bufferType.Describe().c_str(),
                singleRankMemLayout[index].startAddr,
                curBlockEndAddr,
                singleRankMemLayout[index].size);
            return HcclResult::HCCL_E_MEMORY;
        }

        dataSlice.SetBufferType(singleRankMemLayout[index].bufferType);
        dataSlice.SetOffset(addr - singleRankMemLayout[index].startAddr);
        dataSlice.SetSize(len);
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_ERROR("fail to get dataSlice, addr= %p", addr);
    return HcclResult::HCCL_E_MEMORY;
}

HcclResult MemLayout::GetSlice(BufferType bufferType, u64 dstOffset, u64 len, DataSlice& dataSlice, RankId* rank)
{
    dataSlice.SetBufferType(bufferType);
    dataSlice.SetOffset(dstOffset);
    dataSlice.SetSize(len);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemLayout::AivGetSlice(char_t *addr, u64 size, DataSlice &dataSlice, RankId *rank, BlockId *block)
{
    if (addr == 0) {
        HCCL_ERROR("NULL addr input.");
        return HcclResult::HCCL_E_MEMORY;
    }

    RankId curRank = GetRankIdByAddr(addr);
    if (rank != nullptr) {
        *rank = curRank;
    }

    u32 blockId = GetBlockIdByAddr(addr);
    if (block != nullptr) {
        *block = blockId;
    }

    u64 curBlockEndAddr = (u64)allRanksUBMemLayout[curRank][blockId].startAddr + allRanksUBMemLayout[curRank][blockId].size;
    if ((u64)addr + size > curBlockEndAddr) {
        HCCL_ERROR("addr[%p], size[%llu] exceed mem bufferType[%s]: startAddr[%p], endAddr[%p], size[%llu]",
            addr,
            size,
            allRanksUBMemLayout[curRank][blockId].bufferType.Describe().c_str(),
            allRanksUBMemLayout[curRank][blockId].startAddr,
            curBlockEndAddr,
            allRanksUBMemLayout[curRank][blockId].size);
        return HcclResult::HCCL_E_MEMORY;
    }

    dataSlice.SetBufferType(allRanksUBMemLayout[curRank][blockId].bufferType);
    dataSlice.SetOffset((u64)addr - (u64)allRanksUBMemLayout[curRank][0].startAddr);
    dataSlice.SetSize(size);
    return HcclResult::HCCL_SUCCESS;
}

u32 MemLayout::GetMemBlockIdx(BufferType bufferType)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout& singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    for (u32 index = 0; index < singleRankMemLayout.size(); index++) {
        if (singleRankMemLayout[index].bufferType == bufferType) {
            return index;
        }
    }
    // you should never arrive here
    return singleRankMemLayout.size();
}

MemBlock MemLayout::GetMemBlock(BufferType bufferType, RankId curRank)
{
    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout& singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    for (u32 index = 0; index < singleRankMemLayout.size(); index++) {
        if (singleRankMemLayout[index].bufferType == bufferType) {
            return singleRankMemLayout[index];
        }
    }
    return MemBlock{BufferType::INPUT, nullptr, 0};
}

MemBlock MemLayout::GetUBMemBlock(RankId curRank, u32 blockId)
{
    if (allRanksUBMemLayout.find(curRank) == allRanksUBMemLayout.end() ||
        allRanksUBMemLayout[curRank].size() <= blockId) {
        HCCL_ERROR("curRank[%u], blockId[%u] is invalid", curRank, blockId);
        return MemBlock{BufferType::USERBUF_AIV, 0, 0};
    }
    return allRanksUBMemLayout[curRank][blockId];
}

CheckerDataType MemLayout::GetCheckerDataType()
{
    return checkerDataType;
}

u32 MemLayout::GetRankIdByAddr(char_t* addr)
{
    u64 startAddr = (u64)addr & CHECKER_MEM_MASKER;
    return addr2RankId[startAddr];
}

u32 MemLayout::GetBlockIdByAddr(char_t* addr)
{
    u64 startAddr = (u64)addr & AIV_MASKER;
    return addr2RankIdBlockId[startAddr].second;
}

u64 MemLayout::GetBlockMemAddrbyId(RankId curRank, u32 blockId)
{
    char_t *addr = GetUBMemBlock(curRank, blockId).startAddr;
    return (u64)addr & AIV_MASKER;
}

HcclResult MemLayout::SetBufferLen(BufferType bufferType, u64 len)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();

    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout& singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    u32 index = GetMemBlockIdx(bufferType);

    if (len >= CHECKER_MEM_BLOCK_SIZE) {
        HCCL_ERROR("invalid len[%lld]", len);
        return HcclResult::HCCL_E_PARA;
    }

    singleRankMemLayout[index].size = len;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemLayout::SetBlockBufferLen(u32 blockId, u64 len)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();

    if (len >= AIV_MEM_SIZE) {
        HCCL_ERROR("invalid len[%lld]", len);
        return HcclResult::HCCL_E_PARA;
    }

    allRanksUBMemLayout[curRank][blockId].size = len;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemLayout::SetBufferAddrAndLen(BufferType bufferType, char_t* addr, u64 len)
{
    RankId curRank = GetRankIdByAddr(addr);
    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout& singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    u32 index = GetMemBlockIdx(bufferType);

    if (((u64)addr & CHECKER_MEM_MASKER) != ((u64)singleRankMemLayout[index].startAddr & CHECKER_MEM_MASKER)) {
        HCCL_ERROR("new addr[%p] and origin addr[%p] are not in the same block", addr, singleRankMemLayout[index].startAddr);
        return HcclResult::HCCL_E_PARA;
    }

    if(bufferType == BufferType::USERBUF_AIV){
        u64 tailAddr = (u64)addr - ((u64)addr & AIV_MASKER);
        if (tailAddr + len >= AIV_MEM_SIZE) {
            HCCL_ERROR("invalid addr[%p] and  len[%lld]", addr, len);
            return HcclResult::HCCL_E_PARA;
        }
        u32 blockId = GetBlockIdByAddr(addr);

        allRanksUBMemLayout[curRank][blockId].startAddr = addr;
        allRanksUBMemLayout[curRank][blockId].size = len;
        return HcclResult::HCCL_SUCCESS;
    }

    u64 tailAddr = (u64)addr - ((u64)addr & CHECKER_MEM_MASKER);
    if (tailAddr + len >= CHECKER_MEM_BLOCK_SIZE) {
        HCCL_ERROR("invalid addr[%p] and  len[%lld]", addr, len);
        return HcclResult::HCCL_E_PARA;
    }

    singleRankMemLayout[index].startAddr = addr;
    singleRankMemLayout[index].size = len;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemLayout::SetGlobalBuffer(char_t* addr, u64 len)
{
    HCCL_DEBUG("SetGlobalBuffer input addr[%llx] and  len[%lld]", addr, len);
    u64 tailAddr = (u64)addr - ((u64)addr & CHECKER_MEM_MASKER);
    if (tailAddr + len >= CHECKER_MEM_BLOCK_SIZE) {
        HCCL_ERROR("invalid addr[%p] and  len[%lld]", addr, len);
        return HcclResult::HCCL_E_PARA;
    }

    RankId curRank = GetRankIdByAddr(addr);
    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout& singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    for (u32 index = 0; index < singleRankMemLayout.size(); index++) {
        if (((u64)singleRankMemLayout[index].startAddr & CHECKER_MEM_MASKER) == ((u64)addr & CHECKER_MEM_MASKER)) {
            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("SetGlobalBuffer failed.");
    return HcclResult::HCCL_E_PARA;
}

void MemLayout::SetCheckerDataType(CheckerOpParam &opParam)
{

    switch (opParam.opType) {
        case CheckerOpType::ALLTOALLV:
        case CheckerOpType::ALLTOALLVC:
        case CheckerOpType::ALLTOALL:
            checkerDataType = opParam.All2AllDataDes.sendType;
            return;
        case CheckerOpType::REDUCE_SCATTER_V:
        case CheckerOpType::ALLGATHER_V:
            checkerDataType = opParam.VDataDes.dataType;
            return;
        default:
            checkerDataType = opParam.DataDes.dataType;
    }
}

void MemLayout::InitBlockMem(u32 blockNum)
{
    if (hasInitUB || !hasInit) {
        return;
    }
    for (auto curRank = 0; curRank < RankInfoRecorder::Global()->rankSize_; curRank++) {
        u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
        u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
        u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
        if (allSuperPodLayout[superPodId][serverId][rankId].size() <= BufferType::USERBUF_AIV) {
            HCCL_DEBUG("Fail to get startAddr.");
            continue;
        }
        u64 baseAddr = (u64)allSuperPodLayout[superPodId][serverId][rankId][BufferType::USERBUF_AIV].startAddr;
        GenInitUBLayout(curRank, baseAddr, blockNum);
    }
    HCCL_DEBUG("BlockNum is set to [%u]", blockNum);
    hasInitUB = true;
    return;
}

HcclResult MemLayout::TpipeInit(void *&startPtr, void *&endPtr, u32 blockId)
{
    HCCL_DEBUG("=====pipe blockId:%d=====", blockId);
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    u64 startAddr = GetBlockMemAddrbyId(curRank, blockId);
    if (startAddr == 0) {
        HCCL_ERROR("fail to get startAddr");
        return HcclResult::HCCL_E_PARA;
    }
    startPtr = (void *)startAddr;
    endPtr = (void *)(startAddr + CHECKER_MEM_BLOCK_SIZE);
    allRanksUBMemLayout[curRank][blockId].startAddr = (char_t *)startAddr;
    allRanksUBMemLayout[curRank][blockId].size = CHECKER_MEM_BLOCK_SIZE;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemLayout::MemAlloc(u64 simAddr, u64 size)
{
    char *mem = new (std::nothrow) char[size];
    if (mem == nullptr) {
        return HcclResult::HCCL_E_PARA;
    }
    RealMemBlock tmpMem{simAddr, (u64)mem, size};
    simAddr2RealAddr[simAddr] = tmpMem;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemLayout::GetRealAddr(u64 simAddr, u64 &realAddr, u64 &size)
{
    auto it = simAddr2RealAddr.find(simAddr);
    if (it != simAddr2RealAddr.end()) {
        realAddr = it->second.realAddr;
        size = it->second.size;
        return HcclResult::HCCL_SUCCESS;
    }
    for (auto &it : simAddr2RealAddr) {
        if (simAddr > it.first && it.first + it.second.size > simAddr) {
            realAddr = it.second.realAddr + (simAddr - it.first);
            size = it.second.size - (simAddr - it.first);
            return HcclResult::HCCL_SUCCESS;
        }
    }
    return HcclResult::HCCL_E_PARA;
}

BufferType MemLayout::GetBufferType(u64 addr)
{
    RankId curRank = GetRankIdByAddr((char *)addr);
    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
    u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
    SingleRankMemLayout &singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];

    for (u32 index = 0; index < singleRankMemLayout.size(); index++) {
        if (addr < (u64)singleRankMemLayout[index].startAddr) {
            continue;
        }
        u64 curBlockEndAddr = (u64)(singleRankMemLayout[index].startAddr) + singleRankMemLayout[index].size;
        if (curBlockEndAddr < (u64)addr) {
            continue;
        }
        return singleRankMemLayout[index].bufferType;
    }
    return BufferType::RESERVED;
}

HcclResult MemLayout::GenInitUBLayout(RankId rankId, u64 baseAddr, u32 blockNum)
{
    HCCL_DEBUG("=====init ub layout rankID:%d, addr:%llx=====", rankId , baseAddr);
    u64 startAddr = (u64)baseAddr & AIV_MASKER;
    allRanksUBMemLayout[rankId].clear();
    for(int i = 0; i < blockNum; ++i){
        addr2RankIdBlockId[startAddr] = std::make_pair(rankId, i);
        MemBlock inputMemBlock{BufferType::USERBUF_AIV, (char_t *)startAddr, (u64)0};
        allRanksUBMemLayout[rankId].push_back(inputMemBlock);
        startAddr += AIV_MEM_SIZE;
    }
    return HcclResult::HCCL_SUCCESS;
}

SingleRankMemLayout MemLayout::GenInitLayout(RankId rankId)
{
    SingleRankMemLayout initLayout;
    u64 startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;

    addr2RankId[startAddr] = rankId;
    MemBlock inputMemBlock{BufferType::INPUT, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(inputMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock outputMemBlock{BufferType::OUTPUT, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(outputMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock inputCclMemBlock{BufferType::INPUT_CCL, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(inputCclMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock outputCclMemBlock{BufferType::OUTPUT_CCL, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(outputCclMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock scratchMemBlock{BufferType::SCRATCH, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(scratchMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock inputAivMemBlock{BufferType::INPUT_AIV, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(inputAivMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock outputAivMemBlock{BufferType::OUTPUT_AIV, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(outputAivMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock aivCommInfoMemBlock{BufferType::AIV_COMMINFO, (char_t *)checkerMemAddrBase, (u64)0};
    initLayout.push_back(aivCommInfoMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;

    startAddr = (u64)checkerMemAddrBase & CHECKER_MEM_MASKER;
    addr2RankId[startAddr] = rankId;
    MemBlock aivUBMemBlock{BufferType::USERBUF_AIV, (char_t *)checkerMemAddrBase, CHECKER_MEM_BLOCK_SIZE};
    initLayout.push_back(aivUBMemBlock);
    checkerMemAddrBase += CHECKER_MEM_BLOCK_SIZE;
    return initLayout;
}

void MemLayout::Reset()
{
    allSuperPodLayout.clear();
    allRanksUBMemLayout.clear();
    addr2RankIdBlockId.clear();
    for (auto &it : simAddr2RealAddr) {
        delete[] (char *)it.second.realAddr;
    }
    simAddr2RealAddr.clear();
    addr2RankId.clear();
    hasInitUB = false;
    hasInit = false;
    checkerMemAddrBase = CHECKER_MEM_BLOCK_SIZE;
    return;
}

void MemLayout::PrintUB()
{
    for (auto it : allRanksUBMemLayout) {
        HCCL_ERROR("rankId[%u], aivNum[%u]", it.first, it.second.size());
        for (auto aiv : it.second) {
            HCCL_ERROR("aiv startAddr[%llx],size[%llu]", aiv.startAddr, aiv.size);
        }
    }
}

void MemLayout::Init(CheckerOpParam &opParam)
{
    u32 rankSize = RankInfoRecorder::Global()->GetRankSize();
    for (int curRank = 0; curRank < rankSize; curRank++) {
        u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[curRank];
        u32 serverId = RankInfoRecorder::Global()->rankId2serverId[curRank];
        u32 rankId = RankInfoRecorder::Global()->rankId2phyId[curRank];
        allSuperPodLayout[superPodId][serverId][rankId] = GenInitLayout(curRank);

        SingleRankMemLayout &singleRankMemLayout = allSuperPodLayout[superPodId][serverId][rankId];
    }
    hasInit = true;
    SetCheckerDataType(opParam);
    return;
}
}
