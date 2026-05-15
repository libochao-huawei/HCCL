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
#include <array>

#include "log.h"
#include "ccu_api.hpp"
#include "ccu_log.h"
// #include "ccu_kernel.h"

namespace ops_hccl {

constexpr uint64_t CCU_MS_INTERLEAVE         = 8;
constexpr uint64_t CCU_MS_DEFAULT_LOOP_COUNT = 64;
constexpr uint64_t CCU_MS_SIZE               = 4096;

// /* hccl仓CcuKernel基类，提供group高阶操作接口 */
// class CcuKernelAlgBase : public CcuKernel {
// public:
//     // 继承所有构造函数
//     using CcuKernel::CcuKernel;

// protected:
//     // 编程接口
    struct LoopGroupConfig {
        uint32_t msInterleave;  // loop使用的ms步长，即与前一个loop间的间距
        uint32_t loopCount;     // loop的并行次数
        uint64_t memSlice;      // 单个loop内使用的ms总字节大小
    };

    struct LoopGroupResource {
        ccu::Event  completedEvent[CCU_MS_DEFAULT_LOOP_COUNT];
        ccu::CcuBuffer ccuBuf[CCU_MS_DEFAULT_LOOP_COUNT * CCU_MS_INTERLEAVE];
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
        // GroupOpSizeVars goSize;

        LoopGroupConfig  moConfig;
        LoopGroupResource moRes;
        bool resourceAllocated;

        CcuLoop loops[2];
        CcuLoopExecutors enginePool;
        bool loopRegistered;

        // // Loop body 中的外部 LocalAddr（每个 loop index 各两组）
        // ccu::LocalAddr loopDst[2];
        // ccu::LocalAddr loopSrc[2];
        // ccu::LocalAddr loopScratch[2][CCU_MAX_RANK_SIZE];
        // ccu::Variable  loopLen[2];
        // ccu::Variable  loopLenExp[2];
    };

    struct GroupReduceVar {
        ccu::LocalAddr loopDst[2];
        std::array<std::vector<ccu::RemoteAddr>, 2> loopRemoteSrc;
        ccu::LocalAddr loopLocalSrc[2];
        ccu::Variable  loopLen[2];
        ccu::Variable  loopLenExp[2];
    };

//     // 用于n和p部分数据loopgroup的参数
//     GroupOpConfig       moConfig{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
//     GroupOpSizeResource moRes;

//     const uint32_t LOCAL_COPY_MS_PER_LOOP = 8;
//     const uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

//     // 引入基类Load函数，防止名称遮蔽
//     using CcuKernel::LocalReduceNb;
//     // 封装vector接口
//     HcclResult LocalReduceNb(const std::vector<CcuRep::CcuBuf> &bufs, uint32_t count, HcclDataType dataType,
//                      HcclDataType outputDataType, HcclReduceOp opType,
//                      const CcuRep::Variable &len, CcuRep::CompletedEvent event);

//     // loopgroup
//     GroupOpSize CreateGroupOpSize();
//     std::vector<CcuRep::CcuBuf> CreateBlockCcuBuf(uint32_t count);
//     std::vector<CcuRep::Executor> CreateBlockExecutor(uint32_t count);
//     std::vector<CcuRep::CompletedEvent> CreateBlockCompletedEvent(uint32_t count);

//     void LoopGroup(const std::vector<CcuRep::LoopCall> &loops, const std::vector<CcuRep::Variable> &loopCfg,
//         const CcuRep::Variable &paraCfg, const CcuRep::Variable &offsetCfg);

//     // 高阶操作
//     std::vector<uint64_t> CalGoSize(uint64_t size);
    std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig &config);
    CcuResult AllocGoResource(LoopGroupConfig &config, LoopGroupResource &res,
        bool &allocated, uint32_t parallelDim = CCU_MS_DEFAULT_LOOP_COUNT, uint32_t msPerLoop = 1);
//     // 引入基类Load函数，防止名称遮蔽
//     using CcuKernel::Load;
//     void Load(GroupOpSize moSize);

//     HcclResult GroupBroadcast(const std::vector<ChannelHandle>& channels, std::vector<CcuRep::RemoteAddr> dst,
//                               CcuRep::LocalAddr src, GroupOpSize goSize);
//     HcclResult GroupBroadcastWithoutMyRank(const std::vector<ChannelHandle>& channels, std::vector<CcuRep::RemoteAddr> dst,
//                               CcuRep::LocalAddr src, GroupOpSize goSize);
    CcuResult GroupReduce(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                           ccu::LocalAddr dst, std::vector<ccu::RemoteAddr> src, ccu::LocalAddr localSrc,
                           GroupOpSizeVars goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);

//     HcclResult GroupReduceWithoutMyRank(const std::vector<ChannelHandle> &ccuChannels, CcuRep::LocalAddr dst,
//                              std::vector<CcuRep::RemoteAddr> src, GroupOpSize goSize, HcclDataType dataType,
//                              HcclDataType outputDataType, HcclReduceOp opType);

//     HcclResult GroupCopy(CcuRep::LocalAddr dst, CcuRep::LocalAddr src, GroupOpSize goSize);
//     HcclResult GroupLocalReduce(CcuRep::LocalAddr outDstOrg, std::vector<CcuRep::LocalAddr> &scratchOrg,
//         GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
// private:
//     HcclResult CreateMultiOpCopy();
//     HcclResult CreateMultiOpBroadcast(const std::vector<ChannelHandle> &channels);
//     HcclResult CreateMultiOpBroadcastWithoutMyRank(const std::vector<ChannelHandle> &channels);
    CcuResult CreateMultiOpReduce(CcuKernelCtxBase &ctx, GroupReduceVar &var,
                                   const size_t channels[], uint32_t channelCount, HcclDataType dataType,
                                   HcclDataType outputDataType, HcclReduceOp opType);
//     HcclResult CreateMultiOpReduceWithoutMyRank(const std::vector<ChannelHandle> &ccuChannels, HcclDataType dataType,
//                                      HcclDataType outputDataType, HcclReduceOp opType);
//     HcclResult CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
//         HcclReduceOp opType);
//     std::string GetLoopBlockTag(std::string loopType, int32_t index);
// };

}

#endif // !CCU_KERNEL_ALG_BASE
