/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_KERNEL_ALG_BASE
#define CCU_KERNEL_ALG_BASE

#include <vector>
#include <map>
#include <array>
#include <memory>

#include "log.h"
#include "ccu_api.hpp"
#include "ccu_log.h"

namespace ops_hccl {

constexpr uint64_t CCU_MS_INTERLEAVE         = 8;
constexpr uint64_t CCU_MS_DEFAULT_LOOP_COUNT = 64;
constexpr uint64_t CCU_MS_SIZE               = 4096;

struct LoopGroupConfig {
    uint32_t msInterleave;  // loop使用的ms步长，即与前一个loop间的间距
    uint32_t loopCount;     // loop的并行次数
    uint64_t memSlice;      // 单个loop内使用的ms总字节大小
};

struct LoopGroupResource {
    ccu::Array<ccu::Event>     completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t  eventCount;
    uint32_t  bufCount;
};

struct GroupOpSizeVars {
    ccu::Variable addrOffset;        // 第二个loopGroup搬运的起始偏移
    ccu::Variable loopParam;         // loop串行重复执行次数
    ccu::Variable parallelParam;     // loopgroup展开参数，包括展开次数、从第几个loop开始展开、共有几个loop
    ccu::Variable residual;          // 尾块数据size
};

struct CcuKernelCtxBase {
    struct CcuLoopEntity {
        std::unique_ptr<ccu::Func> body[2];
        std::unique_ptr<ccu::Loop> loops[2];
        ccu::Variable              loopParam[2];
    };

    LoopGroupConfig  moConfig;
    LoopGroupResource moRes;
    bool resourceAllocated;

    std::map<std::string, CcuLoopEntity> loopMap;
    CcuLoopExecutors enginePool;

    void CreateLoopEntity(std::string loopStr) {
        loopMap.emplace(loopStr, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(std::string loopStr) {
        return loopMap.count(loopStr) != 0;
    }
};

struct GroupReduceVar {
    ccu::LocalAddr loopDst[2];
    std::array<std::vector<ccu::RemoteAddr>, 2> loopRemoteSrc;
    ccu::LocalAddr loopLocalSrc[2];
    ccu::Variable  loopLen[2];
    ccu::Variable  loopLenExp[2];
};

struct GroupBroadcastVar {
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopLocalDst[2];
    std::array<std::vector<ccu::RemoteAddr>, 2> loopRemoteDst;
    ccu::Variable  loopLen[2];
};

std::vector<uint64_t> CalGoSize(uint64_t size);
std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig &config);
CcuResult AllocGoResource(LoopGroupConfig &config, LoopGroupResource &res,
    bool &allocated, uint32_t parallelDim = CCU_MS_DEFAULT_LOOP_COUNT, uint32_t msPerLoop = 1);

HcclResult GroupBroadcastWithoutMyRank(const std::vector<ChannelHandle>& channels, std::vector<CcuRep::RemoteAddr> dst,
                          CcuRep::LocalAddr src, GroupOpSize goSize);

HcclResult GroupReduceWithoutMyRank(const std::vector<ChannelHandle> &ccuChannels, CcuRep::LocalAddr dst,
                         std::vector<CcuRep::RemoteAddr> src, GroupOpSize goSize, HcclDataType dataType,
                         HcclDataType outputDataType, HcclReduceOp opType);

HcclResult GroupCopy(CcuRep::LocalAddr dst, CcuRep::LocalAddr src, GroupOpSize goSize);
HcclResult GroupLocalReduce(CcuRep::LocalAddr outDstOrg, std::vector<CcuRep::LocalAddr> &scratchOrg,
    GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);

HcclResult CreateMultiOpBroadcastWithoutMyRank(const std::vector<ChannelHandle> &channels);
CcuResult GroupReduce(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                        ccu::LocalAddr dst, std::vector<ccu::RemoteAddr> src, ccu::LocalAddr localSrc,
                        GroupOpSizeVars goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
CcuResult CreateMultiOpReduce(CcuKernelCtxBase &ctx, GroupReduceVar &var,
                               const size_t channels[], uint32_t channelCount, HcclDataType dataType,
                               HcclDataType outputDataType, HcclReduceOp opType);
CcuResult CreateMultiOpBroadcast(CcuKernelCtxBase &ctx, GroupBroadcastVar &var,
                                 const size_t channels[], uint32_t channelCount);
CcuResult GroupBroadcast(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                         ccu::LocalAddr localDst, std::vector<ccu::RemoteAddr> dst, ccu::LocalAddr src, GroupOpSizeVars goSize);
CcuResult CreateMultiOpReduceWithoutMyRank(const std::vector<ChannelHandle> &ccuChannels, HcclDataType dataType,
                                 HcclDataType outputDataType, HcclReduceOp opType);

std::string GetLoopBlockTag(std::string loopType, int32_t index);

}

#endif // !CCU_KERNEL_ALG_BASE