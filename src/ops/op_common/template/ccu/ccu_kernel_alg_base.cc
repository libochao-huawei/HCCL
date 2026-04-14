/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

// HcclResult CcuKernelAlgBase::LocalReduceNb(const std::vector<CcuRep::CcuBuf> &bufs, uint32_t count, HcclDataType dataType,
//                      HcclDataType outputDataType, HcclReduceOp opType,
//                      const CcuRep::Variable &len, CcuRep::CompletedEvent event)
// {
//     (void)count;
//     return CcuKernel::LocalReduceNb(bufs.data(), bufs.size(), dataType, outputDataType, opType, len, event);
// }

CcuResult AllocGoResource(LoopGroupConfig &config, LoopGroupResource &res, bool &allocated, uint32_t parallelDim, uint32_t msPerLoop)
{
    if (allocated) {
        return CCU_SUCCESS;
    }

    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = parallelDim;
    config.memSlice     = msPerLoop * CCU_MS_SIZE;

    res.eventCount = config.loopCount;
    CCU_CHK_RET(ccu::BlockAlloc(res.completedEvent, res.eventCount));

    res.bufCount = config.loopCount * config.msInterleave;
    CCU_CHK_RET(ccu::BlockAlloc(res.ccuBuf, res.bufCount));

    allocated = true;
    return CCU_SUCCESS;
}

// void CcuKernelAlgBase::Load(GroupOpSize moSize)
// {
//     Load(moSize.addrOffset);
//     Load(moSize.loopParam);
//     Load(moSize.parallelParam);
//     Load(moSize.residual);
// }


std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig &config)
{
    uint64_t loopSize = config.loopCount * config.memSlice;     // loop展开后，一次并行搬运的size
    uint64_t maxSize  = loopSize * (GetMaxLoopIterNum() + 1);

    uint64_t m = size / loopSize;                               // 能被loopSize(所有loop并行一次的大小)整除的部分。m为第一个loopgroup展开后的loop串行次数。
    uint64_t n = (size - m * loopSize) / config.memSlice;       // 能被一个loop搬运大小整除的部分。n为第二个loopgroup的展开后的并行loop数。
    uint64_t p = size - m * loopSize - n * config.memSlice;     // 尾数据。p为单独搬运的数据大小。

    if (size == maxSize) {
        m = GetMaxLoopIterNum();
        n = config.loopCount - 1;
        p = config.memSlice;
    }

    HCCL_INFO("Ccu Slice Split: m = %lu, n = %lu, p = %lu", m, n, p);

    // 数据量 < 256K, 跳过LoopGroup0
    // 此时loopIterNum == 0
    // 可以以此做为跳过LoopGroup0的条件
    uint64_t offset      = config.memSlice * config.loopCount * m;
    uint64_t loopIterNum = m;

    uint64_t loopExtendNum = 0;
    uint64_t tailSize      = 0;

    if (n == 0 && p == 0) {
        // 数据量为256K的整数倍，跳过LoopGroup1
        // 此时tailSize = 0，可以依次做为跳过LoopGroup1的条件
        loopExtendNum = 0; // loopExtendNum 赋值
        tailSize      = 0; // tailSize 赋值
    } else if (n != 0 && p == 0) {
        // 数据量为256K * m + 4K * n
        // 因为p == 0, 所以只需要使用第一个Loop, 数据量4K, 展开成n次
        loopExtendNum = GetParallelParam(n - 1, 0, 1); // loopExtendNum 赋值
        tailSize      = config.memSlice;                     // tailSize 赋值
    } else if (n == 0 && p != 0) {
        // 数据量为256K * m + p
        // 因为n == 0, 所以只需要使用第一个Loop, 数据量p, 不展开
        loopExtendNum = GetParallelParam(0, 0, 1); // loopExtendNum 赋值
        tailSize      = p;                                 // tailSize 赋值
    } else {
        loopExtendNum = GetParallelParam(n - 1, 1, 2); // loopExtendNum 赋值, 为2
        tailSize      = p;                                     // tailSize 赋值
    }

    HCCL_INFO("[CalGoSize] offset = %lu, loopIterNum = %lu, loopExtendNum = %lu, tailSize = %lu", offset, loopIterNum,
               loopExtendNum, tailSize);

    return {offset, loopIterNum, loopExtendNum, tailSize};
}

// HcclResult CcuKernelAlgBase::CreateMultiOpBroadcast(const std::vector<ChannelHandle> &channels)
// {
//     AllocGoResource();

//     std::string loopType = "broadcast";
//     if (registeredLoop.find(loopType) != registeredLoop.end()) {
//         return HCCL_SUCCESS;
//     }

//     uint32_t channelSize = channels.size();
//     uint32_t size = channelSize + 1;

//     for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
//         CcuRep::LocalAddr src = CreateLocalAddr();
//         std::vector<CcuRep::RemoteAddr> dst;
//         for (uint32_t i = 0; i < size; i++) {
//             CcuRep::LocalAddr tmp = CreateLocalAddr();
//             dst.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr*>(&tmp));
//         }
//         CcuRep::Variable            len = CreateVariable();
//         CcuRep::LoopBlock           lb(this, loopType + "_loop_" + std::to_string(index));
//         lb(src, dst, len);

//         CcuRep::CcuBuf &buf = moRes.ccuBuf[index * moConfig.msInterleave];
//         CcuRep::CompletedEvent &event = moRes.completedEvent[index];

//         event.mask = 1;
//         LocalCopyNb(buf, src, len, event);
//         WaitEvent(event);

//         for (uint32_t i = 0; i < channels.size(); i++) {
//             if (channels[i] == 0) {
//                 return HCCL_E_PTR;
//             }
//             event.mask = 1 << i;
//             CHK_RET(WriteNb(channels[i], dst[i], buf, len, event));
//         }
//         CcuRep::LocalAddr &localDst = *reinterpret_cast<CcuRep::LocalAddr*>(&dst[size - 1]);
//         event.mask = 1 << channelSize;
//         LocalCopyNb(localDst, buf, len, event);
//         event.mask = (1 << size) - 1;
//         WaitEvent(event);
//     }

//     registeredLoop.insert(loopType);
//     return HCCL_SUCCESS;
// }

// HcclResult CcuKernelAlgBase::GroupBroadcast(const std::vector<ChannelHandle> &channels, std::vector<CcuRep::RemoteAddr> dst,
//                                 CcuRep::LocalAddr src, GroupOpSize goSize)
// {
//     CHK_RET(CreateMultiOpBroadcast(channels));

//     uint32_t size = channels.size() + 1;

//     CCU_IF(goSize.addrOffset != 0)
//     {
//         CcuRep::Variable loopParam = CreateVariable();
//         loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
//         loopParam += goSize.loopParam;

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize = moConfig.memSlice;
//         auto lc   = Loop("broadcast_loop_0")(src, dst, sliceSize);

//         CcuRep::Variable paraCfg = CreateVariable();
//         paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
// #ifdef CcuProfiling
//         std::string groupOpSize = "GroupBroadcast";
//         GroupInfoTemp groupInfo {
//             goSize.loopParam.Id(),
//             goSize.parallelParam.Id(),
//             goSize.residual.Id()
//         };
//         AddCcuProfiling(groupInfo, channels, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED, groupOpSize);
// #endif
//     }

//     CCU_IF(goSize.parallelParam != 0)
//     {
//         src.addr += goSize.addrOffset;
//         for (uint32_t i = 0; i < size; i++) {
//             dst[i].addr += goSize.addrOffset;
//         }

//         auto lc0 = Loop("broadcast_loop_0")(src, dst, goSize.residual);

//         src.addr += goSize.residual;
//         for (uint32_t i = 0; i < size; i++) {
//             dst[i].addr += goSize.residual;
//         }

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize = moConfig.memSlice;
//         auto lc1  = Loop("broadcast_loop_1")(src, dst, sliceSize);

//         CcuRep::Variable loopCfg0 = CreateVariable();
//         loopCfg0 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable loopCfg1 = CreateVariable();
//         loopCfg1 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
// #ifdef CcuProfiling
//         std::string groupOpSize = "GroupBroadcast";
//         GroupInfoTemp groupInfo {
//             goSize.loopParam.Id(),
//             goSize.parallelParam.Id(),
//             goSize.residual.Id()
//         };
//         AddCcuProfiling(groupInfo, channels, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED, groupOpSize);
// #endif
//     }
//     return HCCL_SUCCESS;
// }

CcuResult CreateMultiOpReduce(CcuKernelCtxBase &ctx, GroupReduceVar &var,
                                const size_t channels[], uint32_t channelCount, HcclDataType dataType,
                                     HcclDataType outputDataType, HcclReduceOp opType)
{
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated);

    if (ctx.loopRegistered) {
        return CCU_SUCCESS;
    }

    uint32_t channelSize = channelCount;
    uint32_t size = channelSize + 1;
    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        var.loopRemoteSrc[index].resize(size - 1);
        for (uint32_t i = 0; i < size - 1; i++) {
            // ccu::LocalAddr tmp;
            //CCU_CHK_RET(ccu::Alloc(&tmp));
            // var.loopRemoteSrc[index].emplace_back(*reinterpret_cast<ccu::RemoteAddr*>(&tmp));
            CCU_CHK_RET(ccu::Alloc(&var.loopRemoteSrc[index][i]));
        }
        CCU_CHK_RET(ccu::Alloc(&var.loopLocalSrc[index]));
        CCU_CHK_RET(ccu::Alloc(&var.loopDst[index]));
        CCU_CHK_RET(ccu::Alloc(&var.loopLen[index]));
        CCU_CHK_RET(ccu::Alloc(&var.loopLenExp[index]));

        // std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
        //                                        moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
        
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        CcuEvent loopEvt = ctx.moRes.completedEvent[index];

        CCU_LOOP(ctx.loops[index]) {
            for (uint32_t i = 0; i < channelSize; i++) {
                loopEvt.mask = 1 << i;
                ccu::ReadNb(channels[i], ctx.moRes.ccuBuf[bufBase + i], var.loopRemoteSrc[index][i], var.loopLen[index], loopEvt);
            }

            // ccu::LocalAddr &localSrc = *reinterpret_cast<ccu::LocalAddr*>(&var.loopRemoteSrc[index][channelSize]);
            loopEvt.mask = 1 << channelSize;
            ccu::LocalCopyNb(ctx.moRes.ccuBuf[bufBase + channelSize], var.loopLocalSrc[index], var.loopLen[index], loopEvt);
            loopEvt.mask = (1 << size) - 1;
            ccu::WaitEvent(loopEvt);

            if (size > 1) {
                loopEvt.mask = 1;
                ccu::LocalReduceNb(&ctx.moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType, var.loopLen[index], loopEvt);
                ccu::WaitEvent(loopEvt);
            }

            loopEvt.mask = 1;
            ccu::LocalCopyNb(var.loopDst[index], ctx.moRes.ccuBuf[bufBase], var.loopLenExp[index], loopEvt);
            ccu::WaitEvent(loopEvt);
        }
    }


    ctx.loopRegistered = true;
    return CCU_SUCCESS;
}

// HcclResult CcuKernelAlgBase::CreateMultiOpReduceWithoutMyRank(const std::vector<ChannelHandle> &ccuChannels, HcclDataType dataType,
//                                      HcclDataType outputDataType, HcclReduceOp opType)
// {
//     AllocGoResource();

//     std::string loopType = GetReduceTypeStr(dataType, opType);
//     if (registeredLoop.find(loopType) != registeredLoop.end()) {
//         HCCL_ERROR("registeredLoop.find(loopType) != registeredLoop.end()");
//         return HCCL_SUCCESS;
//     }

//     uint32_t size         = ccuChannels.size();
//     uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
//     uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

//     for (int32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
//         std::vector<CcuRep::RemoteAddr> src;
//         src.reserve(size);
//         for (uint32_t i = 0; i < size; i++) {
//             CcuRep::LocalAddr tmp = CreateLocalAddr();
//             src.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr*>(&tmp));
//         }
//         CcuRep::LocalAddr dst = CreateLocalAddr();
//         CcuRep::Variable  len = CreateVariable();
//         CcuRep::Variable  lenForExpansion = CreateVariable();
//         CcuRep::LoopBlock lb(this, loopType + "_withoutloop_" + std::to_string(index));
//         lb(src, dst, len, lenForExpansion);

//         std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
//                                                moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
//         CcuRep::CompletedEvent &event = moRes.completedEvent[index];
//         for (uint32_t i = 0; i < ccuChannels.size(); i++) {
//             event.mask = 1 << i;
//             ReadNb(ccuChannels[i], bufs[i], src[i], len, event);
//         }

//         event.mask = (1 << size) - 1;
//         WaitEvent(event);

//         if (size > 1) {
//             event.mask = 1;
//             LocalReduceNb(bufs, size, dataType, outputDataType, opType, len, event);
//             WaitEvent(event);
//         }

//         event.mask = 1;
//         LocalCopyNb(dst, bufs[0], lenForExpansion, event);
//         WaitEvent(event);
//     }

//     registeredLoop.insert(loopType);
//     return HCCL_SUCCESS;
// }

// HcclResult CcuKernelAlgBase::CreateMultiOpReduceWithoutMyRank(const std::vector<ChannelHandle> &ccuChannels, HcclDataType dataType,
//                                      HcclDataType outputDataType, HcclReduceOp opType)
// {
//     AllocGoResource();

//     std::string loopType = GetReduceTypeStr(dataType, opType);
//     if (registeredLoop.find(loopType) != registeredLoop.end()) {
//         HCCL_ERROR("registeredLoop.find(loopType) != registeredLoop.end()");
//         return HCCL_SUCCESS;
//     }

//     uint32_t size         = ccuChannels.size();
//     uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
//     uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

//     for (int32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
//         std::vector<CcuRep::RemoteAddr> src;
//         src.reserve(size);
//         for (uint32_t i = 0; i < size; i++) {
//             CcuRep::LocalAddr tmp = CreateLocalAddr();
//             src.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr*>(&tmp));
//         }
//         CcuRep::LocalAddr dst = CreateLocalAddr();
//         CcuRep::Variable  len = CreateVariable();
//         CcuRep::Variable  lenForExpansion = CreateVariable();
//         CcuRep::LoopBlock lb(this, loopType + "_withoutloop_" + std::to_string(index));
//         lb(src, dst, len, lenForExpansion);

//         std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
//                                                moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
//         CcuRep::CompletedEvent &event = moRes.completedEvent[index];
//         for (uint32_t i = 0; i < ccuChannels.size(); i++) {
//             event.mask = 1 << i;
//             ReadNb(ccuChannels[i], bufs[i], src[i], len, event);
//         }

//         event.mask = (1 << size) - 1;
//         WaitEvent(event);

//         if (size > 1) {
//             event.mask = 1;
//             LocalReduceNb(bufs, size, dataType, outputDataType, opType, len, event);
//             WaitEvent(event);
//         }

//         event.mask = 1;
//         LocalCopyNb(dst, bufs[0], lenForExpansion, event);
//         WaitEvent(event);
//     }

//     registeredLoop.insert(loopType);
//     return HCCL_SUCCESS;
// }

CcuResult GroupReduce(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount, ccu::LocalAddr dst,
                        std::vector<ccu::RemoteAddr> src, ccu::LocalAddr localSrc, GroupOpSizeVars goSize, HcclDataType dataType,
                        HcclDataType outputDataType, HcclReduceOp opType)
{
    GroupReduceVar var;
    HCCL_INFO("xjhlog1");
    HCCL_INFO("xjhlog1 ctx.loopRegistered [%d]", (int)ctx.loopRegistered);
    CCU_CHK_RET(CreateMultiOpReduce(ctx, var, channels, channelCount, dataType, outputDataType, opType));


    uint32_t         size         = channelCount + 1;
    uint32_t         expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuVariable sliceSizeExpansion;
    CCU_CHK_RET(ccu::Alloc(&sliceSizeExpansion));
    HCCL_INFO("xjhlog12");

    if (expansionNum != 1) {
        CcuVariable tmp;
        CCU_CHK_RET(ccu::Alloc(&tmp));
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }
    HCCL_INFO("xjhlog13");

    // 第一个loopgroup，只包含1个loop，搬运m部分数据。
    // loopgroup的parallel参数自己生成
    CCU_IF_ONLY(goSize.loopParam != 0)
    {
        CcuVariable loopParam;
        CCU_CHK_RET(ccu::Alloc(&loopParam));
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0); // Load immediate[2147483648] to Xn[23]
        loopParam = loopParam + goSize.loopParam; // Xn[23] + Xn[5] to Xn[23]
        HCCL_INFO("xjhlog14");

        CcuVariable sliceSize;
        CCU_CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;                 // Load immediate[4096] to Xn[24]
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;  // Load immediate[4096] to Xn[21]
        HCCL_INFO("xjhlog15");

        // auto lc = Loop(GetReduceTypeStr(dataType, opType) + "_loop_0")(src, dst, sliceSize, sliceSizeExpansion);
        // 绑定 loop0 的外部 LocalAddr 和 Variable
        for (uint32_t i = 0; i < size - 1; ++i) {
            var.loopRemoteSrc[0][i].addr = src[i].addr;   // GSA[0] + GSA[400] to GSA[0] | GSA[1] + GSA[400] to GSA[0]
            var.loopRemoteSrc[0][i].token = src[i].token; // Xn[8] + Xn[413] to Xn[0] | Xn[9] + Xn[413] to Xn[0]
        }
        HCCL_INFO("xjhlog16");
        var.loopLocalSrc[0].addr = localSrc.addr;
        var.loopLocalSrc[0].token = localSrc.token;
        var.loopDst[0].addr  = dst.addr;         // GSA[2] + GSA[400] to GSA[5]
        var.loopDst[0].token = dst.token;        // Xn[10] + Xn[413] to Xn[13]
        var.loopLen[0]       = sliceSize;          // Xn[24] + Xn[413] to Xn[14]
        var.loopLenExp[0]    = sliceSizeExpansion; // Xn[21] + Xn[413] to Xn[15]

        HCCL_INFO("xjhlog17");
        CcuVariable paraCfg;
        CCU_CHK_RET(ccu::Alloc(&paraCfg));
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);  // Load immediate[2269816411217985536] to Xn[25]

        CcuVariable offsetCfg;
        CCU_CHK_RET(ccu::Alloc(&offsetCfg));
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);  // Load immediate[8589942785] to Xn[26]

        CcuLoopGroup group;
        CCU_CHK_RET(ccu::CreateLoopGroup(&group, &paraCfg, &offsetCfg, ctx.enginePool));
        CCU_CHK_RET(ccu::AddLoop(group, ctx.loops[0], &loopParam));

// #ifdef CcuProfiling
//         std::string groupOpSize = "GroupReduce";
//         GroupInfoTemp groupInfo {
//             goSize.loopParam.Id(),
//             goSize.parallelParam.Id(),
//             goSize.residual.Id()
//         };
//         AddCcuProfiling(groupInfo, channels, dataType, outputDataType, opType, groupOpSize);
// #endif
    }

    // 第二个loopgroup，包含1或2个loop，搬运n和p部分数据。
    // loopgroup的parallel参数使用基类中的moConfig，需要提前调用CalGoSize计算出来。
    CCU_IF_ONLY(goSize.parallelParam != 0)
    {
        for (uint32_t i = 0; i < size - 1; i++) {
            src[i].addr += goSize.addrOffset;  // Load GSA[0] + Xn[4] to GSA[0] | Load GSA[1] + Xn[4] to GSA[1]
        }
        localSrc.addr += goSize.addrOffset;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset; // Load GSA[2] + Xn[4] to GSA[2]
        }

        sliceSizeExpansion = 0;  // immediate[0] to Xn[21]
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + goSize.residual; // Xn[21] + Xn[7] to Xn[21]
        }
        // 绑定 loop0 参数 (p 部分)
        var.loopDst[0].addr  = dst.addr; // GSA[2] + GSA[400] to GSA[5]
        var.loopDst[0].token = dst.token;
        for (uint32_t i = 0; i < size - 1; ++i) {
            var.loopRemoteSrc[0][i].addr = src[i].addr;
            var.loopRemoteSrc[0][i].token = src[i].token;
        }
        var.loopLocalSrc[0].addr = localSrc.addr;
        var.loopLocalSrc[0].token = localSrc.token;
        var.loopLen[0]    = goSize.residual;
        var.loopLenExp[0] = sliceSizeExpansion;

        // auto lc0 = Loop(GetReduceTypeStr(dataType, opType) + "_loop_0")(src, dst, goSize.residual, sliceSizeExpansion);

        for (uint32_t i = 0; i < size - 1; i++) {
            src[i].addr += goSize.residual;
        }
        localSrc.addr += goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuVariable sliceSize;
        CCU_CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        var.loopDst[1].addr  = dst.addr;
        var.loopDst[1].token = dst.token;
        for (uint32_t i = 0; i < size - 1; ++i) {
            var.loopRemoteSrc[1][i].addr = src[i].addr;
            var.loopRemoteSrc[1][i].token = src[i].token;
        }
        var.loopLocalSrc[1].addr = localSrc.addr;
        var.loopLocalSrc[1].token = localSrc.token;
        var.loopLen[1]    = sliceSize;
        var.loopLenExp[1] = sliceSizeExpansion;

        // auto lc1 = Loop(GetReduceTypeStr(dataType, opType) + "_loop_1")(src, dst, sliceSize, sliceSizeExpansion);

        CcuVariable loopCfg0;
        CCU_CHK_RET(ccu::Alloc(&loopCfg0));
        loopCfg0 = GetLoopParam(0, 0, 1);

        CcuVariable loopCfg1;
        CCU_CHK_RET(ccu::Alloc(&loopCfg1));
        loopCfg1 = GetLoopParam(0, 0, 1);

        CcuVariable offsetCfg;
        CCU_CHK_RET(ccu::Alloc(&offsetCfg));
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        CcuLoopGroup group;
        CCU_CHK_RET(ccu::CreateLoopGroup(&group, &goSize.parallelParam, &offsetCfg, ctx.enginePool));
        CCU_CHK_RET(ccu::AddLoop(group, ctx.loops[0], &loopCfg0));
        CCU_CHK_RET(ccu::AddLoop(group, ctx.loops[1], &loopCfg1));
    
//#ifdef CcuProfiling
//        std::string groupOpSize = "GroupReduce";
//        GroupInfoTemp groupInfo {
//            goSize.loopParam.Id(),
//            goSize.parallelParam.Id(),
//            goSize.residual.Id()
//        };
//        AddCcuProfiling(groupInfo, channels, dataType, outputDataType, opType, groupOpSize);
//#endif
    }
    return CCU_SUCCESS;
}

// HcclResult CcuKernelAlgBase::CreateMultiOpBroadcastWithoutMyRank(const std::vector<ChannelHandle> &channels)
// {
//     AllocGoResource();

//     std::string loopType = "broadcast";
//     if (registeredLoop.find(loopType) != registeredLoop.end()) {
//         return HCCL_SUCCESS;
//     }

//     uint32_t size = channels.size() + 1;

//     for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
//         CcuRep::LocalAddr src = CreateLocalAddr();
//         std::vector<CcuRep::RemoteAddr> dst;
//         for (uint32_t i = 0; i < size; i++) {
//             CcuRep::LocalAddr tmp = CreateLocalAddr();
//             dst.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr*>(&tmp));
//         }
//         CcuRep::Variable            len = CreateVariable();
//         CcuRep::LoopBlock           lb(this, loopType + "_loop_" + std::to_string(index));
//         lb(src, dst, len);

//         CcuRep::CcuBuf &buf = moRes.ccuBuf[index * moConfig.msInterleave];
//         CcuRep::CompletedEvent &event = moRes.completedEvent[index];

//         event.mask = 1;
//         LocalCopyNb(buf, src, len, event);
//         WaitEvent(event);

//         for (uint32_t i = 0; i < channels.size(); i++) {
//             if (channels[i] == 0) {
//                 return HCCL_E_PTR;
//             }
//             event.mask = 1 << i;
//             CHK_RET(WriteNb(channels[i], dst[i], buf, len, event));
//         }
//         CcuRep::LocalAddr &localDst = *reinterpret_cast<CcuRep::LocalAddr*>(&dst[size - 1]);
//         event.mask = (1 << (size - 1)) - 1;
//         WaitEvent(event);
//     }

//     registeredLoop.insert(loopType);
//     return HCCL_SUCCESS;
// }

// HcclResult CcuKernelAlgBase::GroupBroadcastWithoutMyRank(const std::vector<ChannelHandle> &channels, std::vector<CcuRep::RemoteAddr> dst,
//                                 CcuRep::LocalAddr src, GroupOpSize goSize)
// {
//     CHK_RET(CreateMultiOpBroadcastWithoutMyRank(channels));

//     uint32_t size = channels.size() + 1;

//     CCU_IF(goSize.addrOffset != 0)
//     {
//         CcuRep::Variable loopParam = CreateVariable();
//         loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
//         loopParam += goSize.loopParam;

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize = moConfig.memSlice;
//         auto lc   = Loop("broadcast_loop_0")(src, dst, sliceSize);

//         CcuRep::Variable paraCfg = CreateVariable();
//         paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
// #ifdef CcuProfiling
//         std::string groupOpSize = "GroupBroadcast";
//         GroupInfoTemp groupInfo {
//             goSize.loopParam.Id(),
//             goSize.parallelParam.Id(),
//             goSize.residual.Id()
//         };
//         AddCcuProfiling(groupInfo, channels, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED, groupOpSize);
// #endif
//     }

//     CCU_IF(goSize.parallelParam != 0)
//     {
//         src.addr += goSize.addrOffset;
//         for (uint32_t i = 0; i < size; i++) {
//             dst[i].addr += goSize.addrOffset;
//         }

//         auto lc0 = Loop("broadcast_loop_0")(src, dst, goSize.residual);

//         src.addr += goSize.residual;
//         for (uint32_t i = 0; i < size; i++) {
//             dst[i].addr += goSize.residual;
//         }

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize = moConfig.memSlice;
//         auto lc1  = Loop("broadcast_loop_1")(src, dst, sliceSize);

//         CcuRep::Variable loopCfg0 = CreateVariable();
//         loopCfg0 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable loopCfg1 = CreateVariable();
//         loopCfg1 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
// #ifdef CcuProfiling
//         std::string groupOpSize = "GroupBroadcast";
//         GroupInfoTemp groupInfo {
//             goSize.loopParam.Id(),
//             goSize.parallelParam.Id(),
//             goSize.residual.Id()
//         };
//         AddCcuProfiling(groupInfo, channels, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED, groupOpSize);
// #endif
//     }
//     return HCCL_SUCCESS;
// }

// HcclResult CcuKernelAlgBase::CreateMultiOpCopy()
// {
//     AllocGoResource(CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP);
//     std::string loopType = "localcopy";
//     if (registeredLoop.find(loopType) != registeredLoop.end()) {
//         return HCCL_SUCCESS;
//     }

//     uint32_t usedBufNum = moConfig.memSlice / CcuRep::CCU_MS_SIZE;

//     for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
//         CcuRep::LocalAddr src = CreateLocalAddr();
//         CcuRep::LocalAddr dst = CreateLocalAddr();
//         CcuRep::Variable  len = CreateVariable();
//         CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
//         lb(src, dst, len);

//         CcuRep::CompletedEvent event = moRes.completedEvent[index];

//         std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
//                                                moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};

//         event.mask = 1;
//         LocalCopyNb(bufs[0], src, len, event);
//         WaitEvent(event);
//         LocalCopyNb(dst, bufs[0], len, event);
//         WaitEvent(event);
//     }

//     registeredLoop.insert(loopType);
//     return HCCL_SUCCESS;
// }

// HcclResult CcuKernelAlgBase::GroupCopy(CcuRep::LocalAddr dst, CcuRep::LocalAddr src, GroupOpSize goSize)
// {
//     CHK_RET(CreateMultiOpCopy());
//     CCU_IF(goSize.addrOffset != 0)
//     {
//         CcuRep::Variable loopParam = CreateVariable();
//         loopParam                  = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
//         loopParam += goSize.loopParam;

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize                  = moConfig.memSlice;
//         auto lc                    = Loop("localcopy_loop_0")(src, dst, sliceSize);

//         CcuRep::Variable paraCfg   = CreateVariable();
//         paraCfg                    = GetParallelParam(moConfig.loopCount - 1, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
//         LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
//     }

//     CCU_IF(goSize.parallelParam != 0)
//     {
//         CcuRep::Condition cond(this, goSize.parallelParam != 0);

//         src.addr += goSize.addrOffset;
//         dst.addr += goSize.addrOffset;
//         auto lc0 = Loop("localcopy_loop_0")(src, dst, goSize.residual);

//         src.addr += goSize.residual;
//         dst.addr += goSize.residual;
//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize                  = moConfig.memSlice;
//         auto lc1                   = Loop("localcopy_loop_1")(src, dst, sliceSize);

//         CcuRep::Variable loopCfg0  = CreateVariable();
//         loopCfg0                   = GetLoopParam(0, 0, 1);
//         CcuRep::Variable loopCfg1  = CreateVariable();
//         loopCfg1                   = GetLoopParam(0, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
//         LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
//     }
//     return HCCL_SUCCESS;
// }

// std::string CcuKernelAlgBase::GetLoopBlockTag(std::string loopType, int32_t index)
// {
//     return loopType + std::to_string(index);
// }

// HcclResult CcuKernelAlgBase::CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
//     HcclReduceOp opType)
// {
//     constexpr uint32_t LOOP_NUM = 16;
//     AllocGoResource(LOOP_NUM);

//     std::string loopType = GetReduceTypeStr(dataType, opType) + "_LocalReduce_Loop_";
//     if (registeredLoop.find(loopType) != registeredLoop.end()) {
//         // 已经注册过
//         return HCCL_SUCCESS;
//     }

//     uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
//     uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;  // ?

//     for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
//         CcuRep::LocalAddr dst = CreateLocalAddr();
//         CcuRep::LocalAddr src = CreateLocalAddr();
//         std::vector<CcuRep::LocalAddr> scratch;
//         for (uint32_t i = 0; i < size; i++) {
//             scratch.emplace_back(CreateLocalAddr());
//         }
//         CcuRep::Variable            len = CreateVariable();
//         CcuRep::Variable            lenForExpansion = CreateVariable();
//         CcuRep::LoopBlock           lb(this, GetLoopBlockTag(loopType, index));
//         lb(dst, scratch, len, lenForExpansion);

//         std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
//                                             moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
//         CcuRep::CompletedEvent     event = moRes.completedEvent[index];

//         for (uint32_t i = 0; i < size; i++) {
//             event.SetMask(1 << i);
//             LocalCopyNb(bufs[i], scratch[i], len, event);
//         }
//         event.SetMask((1 << size) - 1);
//         WaitEvent(event);

//         if (size > 1) {
//             event.SetMask(1);
//             LocalReduceNb(bufs, size, dataType, outputDataType, opType, len, event);
//             WaitEvent(event);
//         }

//         event.SetMask(1);
//         LocalCopyNb(dst, bufs[0], lenForExpansion, event);
//         WaitEvent(event);
//     }

//     registeredLoop.insert(loopType);
//     return HCCL_SUCCESS;
// }

// HcclResult CcuKernelAlgBase::GroupLocalReduce(CcuRep::LocalAddr outDstOrg, std::vector<CcuRep::LocalAddr> &scratchOrg,
//     GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType)
// {
//     const uint32_t size = scratchOrg.size();

//     CcuRep::LocalAddr dst = CreateLocalAddr();
//     dst = outDstOrg;

//     std::vector<CcuRep::LocalAddr> scratch;
//     for (uint32_t idx = 0; idx < size; idx++) {
//         scratch.push_back(CreateLocalAddr());
//         scratch[idx] = scratchOrg[idx];
//     }

//     CreateReduceLoop(size, dataType, outputDataType, opType);

//     std::string loopType = GetReduceTypeStr(dataType, opType) + "_LocalReduce_Loop_";
//     uint32_t         expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
//     CcuRep::Variable sliceSizeExpansion = CreateVariable();

//     if (expansionNum != 1) {
//         CcuRep::Variable tmp = CreateVariable();
//         tmp = GetExpansionParam(expansionNum);
//         dst.token += tmp;
//     }

//     // m部分
//     CCU_IF(goSize.loopParam != 0)                   // goSize1
//     {
//         CcuRep::Variable loopParam = CreateVariable();
//         loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
//         loopParam += goSize.loopParam;

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize          = moConfig.memSlice;
//         sliceSizeExpansion = moConfig.memSlice * expansionNum;

//         auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, scratch, sliceSize, sliceSizeExpansion);

//         CcuRep::Variable paraCfg = CreateVariable();
//         paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
//     }

//     CCU_IF(goSize.parallelParam != 0)               // goSize2
//     {
//         // p部分，加m的偏移
//         for (uint32_t i = 0; i < size; i++) {
//             scratch[i].addr += goSize.addrOffset;
//         }

//         for (uint32_t i = 0; i < expansionNum; i++) {
//             dst.addr += goSize.addrOffset;
//         }

//         sliceSizeExpansion = 0;
//         for (uint32_t i = 0; i < expansionNum; i++) {
//             sliceSizeExpansion += goSize.residual;  // goSize3
//         }

//         auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, scratch, goSize.residual, sliceSizeExpansion);

//         // n部分，再加p的偏移
//         for (uint32_t i = 0; i < size; i++) {
//             scratch[i].addr += goSize.residual;
//         }

//         for (uint32_t i = 0; i < expansionNum; i++) {
//             dst.addr += goSize.residual;
//         }

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize          = moConfig.memSlice;
//         sliceSizeExpansion = moConfig.memSlice * expansionNum;

//         auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, scratch, sliceSize, sliceSizeExpansion);

//         CcuRep::Variable loopCfg0 = CreateVariable();
//         loopCfg0 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable loopCfg1 = CreateVariable();
//         loopCfg1 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
//     }
//     return HCCL_SUCCESS;
// }

// HcclResult CcuKernelAlgBase::GroupReduceWithoutMyRank(const std::vector<ChannelHandle> &ccuChannels, CcuRep::LocalAddr dst,
//                              std::vector<CcuRep::RemoteAddr> src, GroupOpSize goSize, HcclDataType dataType,
//                              HcclDataType outputDataType, HcclReduceOp opType)
// {
//     CHK_RET(CreateMultiOpReduceWithoutMyRank(ccuChannels, dataType, outputDataType, opType));

//     uint32_t         size         = src.size();
//     uint32_t         expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
//     CcuRep::Variable sliceSizeExpansion = CreateVariable();

//     if (expansionNum != 1) {
//         CcuRep::Variable tmp = CreateVariable();
//         tmp = GetExpansionParam(expansionNum);
//         dst.token += tmp;
//     }

//     // 第一个loopgroup，只包含1个loop，搬运m部分数据。
//     // loopgroup的parallel参数自己生成
//     CCU_IF(goSize.loopParam != 0)
//     {
//         CcuRep::Variable loopParam = CreateVariable();
//         loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
//         loopParam += goSize.loopParam;

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize          = moConfig.memSlice;
//         sliceSizeExpansion = moConfig.memSlice * expansionNum;

//         auto lc = Loop(GetReduceTypeStr(dataType, opType) + "_withoutloop_0")(src, dst, sliceSize, sliceSizeExpansion);

//         CcuRep::Variable paraCfg = CreateVariable();
//         paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
// #ifdef CcuProfiling
//         std::string groupOpSize = "GroupReduce";
//         GroupInfoTemp groupInfo {
//             goSize.loopParam.Id(),
//             goSize.parallelParam.Id(),
//             goSize.residual.Id()
//         };
//         AddCcuProfiling(groupInfo, ccuChannels, dataType, outputDataType, opType, groupOpSize);
// #endif
//     }

//     // 第二个loopgroup，包含1或2个loop，搬运n和p部分数据。
//     // loopgroup的parallel参数使用基类中的moConfig，需要提前调用CalGoSize计算出来。
//     CCU_IF(goSize.parallelParam != 0)
//     {
//         for (uint32_t i = 0; i < size; i++) {
//             src[i].addr += goSize.addrOffset;
//         }
//         for (uint32_t i = 0; i < expansionNum; i++) {
//             dst.addr += goSize.addrOffset;
//         }

//         sliceSizeExpansion = 0;
//         for (uint32_t i = 0; i < expansionNum; i++) {
//             sliceSizeExpansion += goSize.residual;
//         }

//         auto lc0 = Loop(GetReduceTypeStr(dataType, opType) + "_withoutloop_0")(src, dst, goSize.residual, sliceSizeExpansion);

//         for (uint32_t i = 0; i < size; i++) {
//             src[i].addr += goSize.residual;
//         }
//         for (uint32_t i = 0; i < expansionNum; i++) {
//             dst.addr += goSize.residual;
//         }

//         CcuRep::Variable sliceSize = CreateVariable();
//         sliceSize          = moConfig.memSlice;
//         sliceSizeExpansion = moConfig.memSlice * expansionNum;

//         auto lc1 = Loop(GetReduceTypeStr(dataType, opType) + "_withoutloop_1")(src, dst, sliceSize, sliceSizeExpansion);

//         CcuRep::Variable loopCfg0 = CreateVariable();
//         loopCfg0 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable loopCfg1 = CreateVariable();
//         loopCfg1 = GetLoopParam(0, 0, 1);
//         CcuRep::Variable offsetCfg = CreateVariable();
//         offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

//         LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
// #ifdef CcuProfiling
//         std::string groupOpSize = "GroupReduce";
//         GroupInfoTemp groupInfo {
//             goSize.loopParam.Id(),
//             goSize.parallelParam.Id(),
//             goSize.residual.Id()
//         };
//         AddCcuProfiling(groupInfo, ccuChannels, dataType, outputDataType, opType, groupOpSize);
// #endif
//     }
//     return HCCL_SUCCESS;
// }

// void CcuKernelAlgBase::LoopGroup(const std::vector<CcuRep::LoopCall> &loops, const std::vector<CcuRep::Variable> &loopCfg,
//                            const CcuRep::Variable &paraCfg, const CcuRep::Variable &offsetCfg)
// {
//     auto                          lgc = CcuRep::LoopGroupCall(this);
//     std::vector<CcuRep::Executor> executors;
//     for (size_t i = 0; i < loops.size(); i++) {
//         executors.push_back(moRes.executor[i]);
//     }
//     lgc.Run(loops, loopCfg, executors, paraCfg, offsetCfg);
// }

// CcuKernelAlgBase::GroupOpSize CcuKernelAlgBase::CreateGroupOpSize()
// {
//     return GroupOpSize{CreateVariable(), CreateVariable(), CreateVariable(), CreateVariable()};
// }

// std::vector<CcuRep::CcuBuf> CcuKernelAlgBase::CreateBlockCcuBuf(uint32_t count)
// {
//     std::vector<CcuRep::CcuBuf> res(count);
//     CcuKernel::CreateBlockCcuBuf(count, res.data());
//     return res;
// }

// std::vector<CcuRep::Executor> CcuKernelAlgBase::CreateBlockExecutor(uint32_t count)
// {
//     std::vector<CcuRep::Executor> res(count);
//     CcuKernel::CreateBlockExecutor(count, res.data());
//     return res;
// }

// std::vector<CcuRep::CompletedEvent> CcuKernelAlgBase::CreateBlockCompletedEvent(uint32_t count)
// {
//     std::vector<CcuRep::CompletedEvent> res(count);
//     CcuKernel::CreateBlockCompletedEvent(count, res.data());
//     return res;
// }

}
