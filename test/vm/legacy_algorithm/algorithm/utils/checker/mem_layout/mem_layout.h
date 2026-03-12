#ifndef HCCLV1_MEM_LAYOUT_H
#define HCCLV1_MEM_LAYOUT_H

#include <vector>
#include <map>

#include "base.h"
#include "checker_data_slice.h"
#include "topo_meta.h"
#include "checker_def.h"

#ifndef HCCL_ALG_ANALYZER_DAVID
#include "mem_device_pub.h"
using namespace hccl;
#endif

namespace checker {

// 每块MemBlock为1T的大小
constexpr u64 CHECKER_MEM_BLOCK_SIZE = 0x10000000000;
constexpr u64 CHECKER_MEM_MASKER = 0xFFFFFF0000000000;

// 每块UB为1G的大小
constexpr u64 AIV_MEM_SIZE = 0x40000000;  // 1G
constexpr u64 AIV_UB_MAX_SIZE = 0x30000;  // UB 192 K
constexpr u64 AIV_MASKER = 0xFFFFFFFFF0000000;

struct MemBlock {
    BufferType bufferType;
    char_t *startAddr;
    u64 size;
};

struct RealMemBlock {
    u64 mulAddr;
    u64 realAddr;
    u64 size;
};

using SingleRankMemLayout = std::vector<MemBlock>;
using AllRankMemLayout = std::map<RankId, SingleRankMemLayout>;
using AllServerMemLayout = std::map<ServerId, AllRankMemLayout>;
using AllSuperPodMemLayout = std::map<SuperPodId, AllServerMemLayout>;

using SingleRankUBLayout = std::vector<MemBlock>;
using AllRanksUBMemLayout = std::map<RankId, SingleRankUBLayout>;

using StartAddr2RankId = std::map<u64, RankId>;
using StartAddr2RankIdBlockId = std::map<u64, std::pair<RankId, BlockId>>;

using SimAddr2RealAddr = std::map<u64, RealMemBlock>;

class MemLayout {
public:
    static MemLayout* Global();
    HcclResult GetSlice(char_t *addr, u64 dataCount, const HcclDataType dataType, DataSlice& dataSlice, RankId* rank = nullptr);
#ifndef HCCL_ALG_ANALYZER_DAVID
    HcclResult GetSlice(const DeviceMem &deviceMem, DataSlice& dataSlice, RankId* rank = nullptr);
#endif
    HcclResult GetSlice(char_t *addr, u64 len, DataSlice& dataSlice, RankId* rank = nullptr);
    HcclResult GetSlice(BufferType bufferType, u64 dstOffset, u64 len, DataSlice& dataSlice, RankId* rank = nullptr);
    HcclResult AivGetSlice(char_t *addr, u64 size, DataSlice &dataSlice, RankId *rank = nullptr, BlockId *aiv = nullptr);

    HcclResult SetBufferLen(BufferType bufferType, u64 len);
    HcclResult SetBlockBufferLen(u32 blockId, u64 len);
    HcclResult SetBufferAddrAndLen(BufferType bufferType, char_t* addr, u64 len);
    HcclResult SetGlobalBuffer(char_t* addr, u64 len);
    void SetCheckerDataType(CheckerOpParam &opParam);
    void InitBlockMem(u32 blockNum);

    HcclResult TpipeInit(void *&startPtr, void *&endPtr, u32 blockId);

    u32 GetRankIdByAddr(char_t* addr);
    u32 GetBlockIdByAddr(char_t* addr);
    u64 GetBlockMemAddrbyId(RankId curRank, u32 blockId);

    HcclResult MemAlloc(u64 simAddr, u64 size);
    HcclResult GetRealAddr(u64 simAddr, u64 &realAddr, u64 &size);
    BufferType GetBufferType(u64 addr);

    void Reset();
    void Init(CheckerOpParam &opParam);
    MemBlock GetMemBlock(BufferType bufferType, RankId curRank);
    MemBlock GetUBMemBlock(RankId curRank, u32 blockId);
    CheckerDataType GetCheckerDataType();

    AllSuperPodMemLayout allSuperPodLayout;
    AllRanksUBMemLayout allRanksUBMemLayout;

    void PrintUB();
private:
    u32 GetMemBlockIdx(BufferType bufferType);
    HcclResult GenInitUBLayout(RankId rankId, u64 baseAddr, u32 blockNum);
    SingleRankMemLayout GenInitLayout(RankId rankId);

    StartAddr2RankId addr2RankId;
    StartAddr2RankIdBlockId addr2RankIdBlockId;
    SimAddr2RealAddr simAddr2RealAddr;

    bool hasInit = false;
    bool hasInitUB = false;
    CheckerDataType checkerDataType = CheckerDataType::DATA_TYPE_RESERVED;
    u64 checkerMemAddrBase = CHECKER_MEM_BLOCK_SIZE;
};
}

#endif
