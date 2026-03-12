#ifndef SIM_CONTEXT_MANAGER_H
#define SIM_CONTEXT_MANAGER_H

#include <unordered_map>
#include <iostream>
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl_mem_defs.h"

namespace HcclProxy {

struct HcclMemHash {
    static constexpr size_t MEM_ADDR_SHIFT = 1;
    static constexpr size_t MEM_SIZE_SHIFT = 2;
    size_t operator()(const HcclMem& mem) const noexcept {
        size_t hashMemType = std::hash<uint32_t>{}(static_cast<int>(mem.type));
        size_t hashMemaddr = std::hash<void*>{}(mem.addr);
        size_t hashMemSize = std::hash<uint64_t>{}(mem.size);
        return hashMemType ^ (hashMemaddr << MEM_ADDR_SHIFT) ^ (hashMemSize << MEM_SIZE_SHIFT);
    }
};

struct HcclMemEqual {
    bool operator()(const HcclMem& lhm, const HcclMem& rhm) const {
        return lhm.type == rhm.type && lhm.addr == rhm.addr && lhm.size == rhm.size;
    }
};

class SimContextMgr {
public:
    SimContextMgr() = default;
    ~SimContextMgr();

    HcclResult CreateCommEngineCtx(const std::string &tag, CommEngine engine, HcclMem *engineCtx);
    HcclResult GetCommEngineCtx(const std::string &tag, CommEngine engine, HcclMem *engineCtx);
    HcclResult CopyCommEngineCtx(const std::string &tag, CommEngine engine, const void *srcCtx,
        uint64_t size, uint64_t dstCtxOffset);
    HcclResult DestroyCommEngineCtx(const HcclMem *engineCtx);

private:
    std::unordered_map<std::string, std::unordered_map<CommEngine, HcclMem>> contextMap_;
    std::unordered_map<HcclMem, std::string, HcclMemHash, HcclMemEqual> tagMap_;
    std::unordered_map<HcclMem, CommEngine, HcclMemHash, HcclMemEqual> engineMap_;
};

}   // namespace HcclSim
#endif // SIM_CONTEXT_MANAGER_H