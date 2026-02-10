/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
// #include "hccl/hccl_res.h"
// #include "sim_communicator.h"
#include "hccl_sim_world_pub.h"
#include "hccl_common_defs.h"
#include <dlfcn.h>
#include "sim_runner_ops.h"
#include "reduce_op.h"
#include "hccl_vm_log.h"
#include "hccl_rank_graph.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include "dtype_common.h"

void print_stacktrace() {
    const int max_frames = 64;
    void* buffer[max_frames];

    int nptrs = backtrace(buffer, max_frames);
    char** strings = backtrace_symbols(buffer, nptrs);

    if (strings == nullptr) {
        perror("backtrace_symbols");
        return;
    }

    fprintf(stderr, "==== Stack trace ====\n");
    for (int i = 0; i < nptrs; i++) {
        fprintf(stderr, "%s\n", strings[i]);
    }
    fprintf(stderr, "=====================\n");

    free(strings);
}

const std::map<HcclDataType, u32> DATA_TYPE_SIZE_MAP = {
    {HcclDataType::HCCL_DATA_TYPE_INT8, sizeof(s8)},
    {HcclDataType::HCCL_DATA_TYPE_INT16, sizeof(s16)},
    {HcclDataType::HCCL_DATA_TYPE_INT32, sizeof(s32)},
    {HcclDataType::HCCL_DATA_TYPE_FP16, 2},
    {HcclDataType::HCCL_DATA_TYPE_FP32, sizeof(float)},
    {HcclDataType::HCCL_DATA_TYPE_INT64, sizeof(s64)},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, sizeof(u64)},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, sizeof(u8)},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, sizeof(u16)},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, sizeof(u32)},
    {HcclDataType::HCCL_DATA_TYPE_FP64, sizeof(double)}, 
    {HcclDataType::HCCL_DATA_TYPE_BFP16, 2},
    {HcclDataType::HCCL_DATA_TYPE_INT128, 16},
    {HcclDataType::HCCL_DATA_TYPE_HIF8, 1},
    {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, 1},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, 1},
    {HcclDataType::HCCL_DATA_TYPE_FP8E8M0, 1}
};

namespace {
    void RegisterBufferType(uint32_t rankId, const void* ptr, BufferType bufType, uint64_t size)
    {
        sim::MemoryLayout memLayout;
        memLayout.rank_id = rankId;
        memLayout.base_addr = reinterpret_cast<uint64_t>(ptr);
        memLayout.buf_type  = bufType;
        memLayout.global_offset = 0; // ?
        memLayout.size = size;
        std::cout<<"[INIT Buffer] rank"<<rankId<<", addr= "<<memLayout.base_addr<<", type= "<<bufType<<", size= "<< memLayout.size<<std::endl;
        auto memKey = RunnerDB::Add<sim::MemoryLayout>(memLayout);
    }
}


using namespace HcclSim;

#ifdef __cplusplus
extern "C" {
#endif
// constexpr uint32_t DATA_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {
//     sizeof(int8_t),     // int8
//     sizeof(int16_t),    // int16
//     sizeof(int32_t),    // int32
//     2,              // fp16
//     sizeof(float),  // fp32
//     sizeof(int64_t),    // int64
//     sizeof(uint64_t),    // uint64
//     sizeof(uint8_t),     // uint8
//     sizeof(uint16_t),    // uint16
//     sizeof(uint32_t),    // uint32
//     8,              // fp64
//     2,              // bfp16
//     16,             // int128
//     1,              // hif8
//     1,              // fp8e4m3
//     1               // fp8e5m2
// };

// uint64_t CalcDataSize(HcclDataType dataType, uint64_t dataCount)
// {
//     if (dataType >= HCCL_DATA_TYPE_RESERVED) {
//         // invalid data type
//         printf("[CalcDataSize] invalid dataType %d\n", dataType);
//         return 0;
//     }
//     uint64_t dataTypeSize = DATA_SIZE_TABLE[dataType];
//     return dataTypeSize * dataCount;
// }

// HcclResult RegisterOpSendRecvBuffer(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType)
// {
//     uint32_t curRank = sim::GetCurrRankId();
//     uint64_t dataSize = CalcDataSize(dataType, recvCount);

//     uint32_t mode = SHMManager::GetHcclVmMode();
//     if (mode == HcclSim::HcclVmMode::CHECKER) {
//         // checker
//         HcclSim::HcclVmResult ret = RegisterNpuMemory(curRank, sendBuf, dataSize, 0);
//         if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
//             printf("[ERROR] [%s] SetIndependentOpConfig fail\n", __func__);
//             return HcclResult::HCCL_E_PARA;
//         }
//         ret = RegisterNpuMemory(curRank, recvBuf, dataSize, 1);
//         if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
//             printf("[ERROR] [%s] SetIndependentOpConfig fail\n", __func__);
//             return HcclResult::HCCL_E_PARA;
//         }
//     }
//     return HcclResult::HCCL_SUCCESS;
// }

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    printf("HcclAlltoAll called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  sendCount = %lu\n", sendCount);
    printf("  recvCount = %lu\n", recvCount);
    printf("  sendType = %d\n", sendType);
    printf("  recvType = %d\n", recvType);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer: zhf-todo: 后续不同算子，此处计算size方式可能不同，如allgather的output buffer
    auto inDataSize = DATA_TYPE_SIZE_MAP.at(sendType);
    auto outDataSize = DATA_TYPE_SIZE_MAP.at(recvType);
    RegisterBufferType(curRank, sendBuf, BufferType::INPUT, inDataSize * sendCount * rankSize);
    RegisterBufferType(curRank, recvBuf, BufferType::OUTPUT, outDataSize * recvCount * rankSize);

    printf("HcclAlltoAll get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALL);
    simData.all2AllDataDes.sendType = static_cast<uint16_t>(sendType);
    simData.all2AllDataDes.recvType = static_cast<uint16_t>(recvType);
    simData.all2AllDataDes.sendCount = sendCount;
    simData.all2AllDataDes.recvCount = recvCount;
    simData.all2AllDataDes.count = rankSize * rankSize;
    std::cout<<"zhf-OP_DEBUG: "<< rankSize<<", "<<simData.all2AllDataDes.count<<std::endl;
    for (uint32_t i = 0; i < rankSize * rankSize; i++) {
        simData.all2AllDataDes.sendCountMatrix[i] = sendCount / rankSize;
        std::cout<<"zhf-OP_DEBUG_PER: "<<sendCount / rankSize<<std::endl;
    }
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);

    using HcclAlltoAllFunc = HcclResult (*)(
        const void *, uint64_t, HcclDataType, const void *, uint64_t, HcclDataType, HcclComm, aclrtStream);
    HcclAlltoAllFunc hcclAlltoAllFunc = reinterpret_cast<HcclAlltoAllFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAlltoAllFunc != nullptr) {
        return hcclAlltoAllFunc(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream)
{
    printf("HcclAlltoAllV called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  sendType = %d\n", sendType);
    printf("  recvType = %d\n", recvType);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer
    auto inDataSize = DATA_TYPE_SIZE_MAP.at(sendType);
    auto outDataSize = DATA_TYPE_SIZE_MAP.at(recvType);
    uint32_t inCountTotal = 0;
    uint32_t outCountTotal = 0;
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        inCountTotal += ((uint64_t*)sendCounts)[rank];
        outCountTotal += ((uint64_t*)recvCounts)[rank];
    }
    RegisterBufferType(curRank, sendBuf, BufferType::INPUT, inDataSize * inCountTotal);
    RegisterBufferType(curRank, recvBuf, BufferType::OUTPUT, outDataSize * outCountTotal);

    printf("HcclAlltoAllV get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV);
    simData.all2AllDataDes.sendType = static_cast<uint16_t>(sendType);
    simData.all2AllDataDes.recvType = static_cast<uint16_t>(recvType);
    simData.all2AllDataDes.count = rankSize * rankSize;
    std::cout<<"zhf-OP_DEBUG: "<< rankSize<<", "<<simData.all2AllDataDes.count<<std::endl;
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        simData.all2AllDataDes.sendCountMatrix[curRank * rankSize + rank] = ((uint64_t*)sendCounts)[rank];
        std::cout<<"zhf-OP_DEBUG_PER: "<<((uint64_t*)sendCounts)[rank]<<std::endl;
    }
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);

    using HcclAlltoAllVFunc = HcclResult (*)(const void *,
        const void *,
        const void *,
        HcclDataType,
        const void *,
        const void *,
        const void *,
        HcclDataType,
        HcclComm,
        aclrtStream);
    HcclAlltoAllVFunc hcclAlltoAllVFunc = reinterpret_cast<HcclAlltoAllVFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAlltoAllVFunc != nullptr) {
        return hcclAlltoAllVFunc(
            sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream)
{
    printf("HcclAllGather called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  sendCount = %lu\n", sendCount);
    printf("  dataType = %d\n", dataType);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer: zhf-todo: 后续不同算子，此处计算size方式可能不同，如allgather的output buffer
    auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    RegisterBufferType(curRank, sendBuf, BufferType::INPUT, dataSize * sendCount);
    RegisterBufferType(curRank, recvBuf, BufferType::OUTPUT, dataSize * sendCount * rankSize);

    printf("HcclAllGather get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    simData.data_type  = static_cast<uint16_t>(dataType);
    simData.data_count = sendCount;
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);
    
    using HcclAllGatherFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclComm, aclrtStream);
    HcclAllGatherFunc hcclAllGatherFunc = reinterpret_cast<HcclAllGatherFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAllGatherFunc != nullptr) {
        return hcclAllGatherFunc(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    printf("HcclBroadcast called with parameters:\n");
    printf("  buf = %p\n", buf);
    printf("  count = %lu\n", count);
    printf("  dataType = %d\n", dataType);
    printf("  root = %u\n", root);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer: zhf-todo: 后续不同算子，此处计算size方式可能不同，如allgather的output buffer
    auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    RegisterBufferType(curRank, buf, BufferType::INPUT, dataSize * count);

    printf("HcclBroadcast get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.root       = root;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_BROADCAST);
    simData.data_type  = static_cast<uint16_t>(dataType);
    simData.data_count = count;
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);
    
    using HcclBroadcastFunc = HcclResult (*)(void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    HcclBroadcastFunc hcclBroadcastFunc = reinterpret_cast<HcclBroadcastFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclBroadcastFunc != nullptr) {
        return hcclBroadcastFunc(buf, count, dataType, root, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    printf("HcclAllReduce called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  count = %lu\n", count);
    printf("  dataType = %d\n", dataType);
    printf("  op = %d\n", op);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer: zhf-todo: 后续不同算子，此处计算size方式可能不同，如allgather的output buffer
    auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    RegisterBufferType(curRank, sendBuf, BufferType::INPUT, dataSize * count);
    RegisterBufferType(curRank, recvBuf, BufferType::OUTPUT, dataSize * count);

    printf("HcclAllReduce get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    simData.data_type  = static_cast<uint16_t>(dataType);
    simData.reduce_op  = static_cast<uint16_t>(op);
    simData.data_count = count;
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);
    
    using HcclAddreduceFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    HcclAddreduceFunc hcclAddreduceFunc = reinterpret_cast<HcclAddreduceFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclAddreduceFunc != nullptr) {
        return hcclAddreduceFunc(sendBuf, recvBuf, count, dataType, op, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    printf("HcclScatter called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  recvCount = %lu\n", recvCount);
    printf("  dataType = %d\n", dataType);
    printf("  root = %u\n", root);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer: zhf-todo: 后续不同算子，此处计算size方式可能不同，如allgather的output buffer
    auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    if (curRank == root) {
        RegisterBufferType(curRank, sendBuf, BufferType::INPUT, dataSize * recvCount * rankSize);
    }
    RegisterBufferType(curRank, recvBuf, BufferType::OUTPUT, dataSize * recvCount);

    printf("HcclScatter get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.root       = root;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_SCATTER);
    simData.data_type  = static_cast<uint16_t>(dataType);
    simData.data_count = recvCount;
    simData.root = root;
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);
    
    using HcclScatterFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    HcclScatterFunc hcclScatterFunc = reinterpret_cast<HcclScatterFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclScatterFunc != nullptr) {
        return hcclScatterFunc(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size)
{
    printf("HcclGetHcclBufferNew called with parameters: buffer= %p, %lu\n", *buffer, *size);
    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    using HcclGetHcclBufferFunc = HcclResult (*)(HcclComm, void**, uint64_t*);
    auto hcclGetHcclBufferFunc = reinterpret_cast<HcclGetHcclBufferFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclGetHcclBufferFunc != nullptr) {
        auto ret = hcclGetHcclBufferFunc(comm, buffer, size);
        RegisterBufferType(curRank, *buffer, BufferType::CCL, *size);
        printf("HcclGetHcclBufferNew get rank%u ccl buffer= %p, %lx\n", curRank, *buffer, *size);
        return ret;
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream)
{
    printf("HcclReduce called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  count = %lu\n", count);
    printf("  dataType = %d\n", dataType);
    printf("  reduce op = %u\n", static_cast<int>(op));
    printf("  root = %d\n", root);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer: zhf-todo: 后续不同算子，此处计算size方式可能不同，如allgather的output buffer
    auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    if (root == curRank) {
        RegisterBufferType(curRank, recvBuf, BufferType::OUTPUT, dataSize * count);
    }
    RegisterBufferType(curRank, sendBuf, BufferType::INPUT, dataSize * count);

    printf("HcclReduce get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_REDUCE);
    simData.reduce_op  = static_cast<uint16_t>(op);
    simData.data_type  = static_cast<uint16_t>(dataType);
    simData.data_count = count;
    simData.root = root;
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);

    using HcclReduceFunc =
        HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
    auto hcclReduceFunc = reinterpret_cast<HcclReduceFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclReduceFunc != nullptr) {
        return hcclReduceFunc(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    printf("HcclReduceScatter called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  recvCount = %lu\n", recvCount);
    printf("  dataType = %d\n", dataType);
    printf("  reduce op = %u\n", static_cast<int>(op));
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();
    uint32_t rankSize = sim::GetRankSize();
    // 注册input、output buffer: zhf-todo: 后续不同算子，此处计算size方式可能不同，如allgather的output buffer
    auto dataSize = DATA_TYPE_SIZE_MAP.at(dataType);
    RegisterBufferType(curRank, sendBuf, BufferType::INPUT, dataSize * recvCount * rankSize);
    RegisterBufferType(curRank, recvBuf, BufferType::OUTPUT, dataSize * recvCount);

    printf("HcclReduceScatter get op info: allRank= %u, curRank= %u.\n", rankSize, curRank);

    // 获取算子信息
    sim::SimModelData simData;
    simData.rank_id    = curRank;
    simData.rank_size  = rankSize;
    simData.chip_type  = static_cast<uint16_t>(DevType::DEV_TYPE_910_95); // zhf-todo: 从DB获取
    simData.op_type    = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    simData.reduce_op  = static_cast<uint16_t>(op);
    simData.data_type  = static_cast<uint16_t>(dataType);
    simData.data_count = recvCount;
    auto simDataKey    = RunnerDB::Add<sim::SimModelData>(simData);
    
    using HcclReduceScatterFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    auto hcclReduceScatterFunc = reinterpret_cast<HcclReduceScatterFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclReduceScatterFunc != nullptr) {
        return hcclReduceScatterFunc(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

// HcclResult HcclCcuKernelRegister(HcclComm comm,
//     CcuKernelHandle *kernelHandle, void *kernelCreator, void *kernelArg)
// {
//     using HcclReduceScatterFunc = HcclResult (*)(HcclComm, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
//     auto hcclReduceScatterFunc = reinterpret_cast<HcclReduceScatterFunc>(dlsym(RTLD_NEXT, __func__));
//     if (hcclReduceScatterFunc != nullptr) {
//         return hcclReduceScatterFunc(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
//     } else {
//         printf("[ERROR] dlsym %s failed\n", __func__);
//         return HcclResult::HCCL_E_NOT_SUPPORT;
//     }
// }

// extern "C" uint16_t _ZN5hcomm6CcuRep15GetUBReduceTypeEN4Hccl8ReduceOpE(Hccl::ReduceOp reduceOp)
// {
//     HCCL_VM_DEBUG("[{}] zhf-GetUBReduceType enter: {:d}", __func__, static_cast<int>(reduceOp));
//     print_stacktrace();
//     using Func = uint16_t (*)(Hccl::ReduceOp);
//     auto func = reinterpret_cast<Func>(dlsym(RTLD_NEXT, __func__));
//     if (func != nullptr) {
//         return func(reduceOp);
//     } else {
//         printf("[ERROR] dlsym %s failed\n", __func__);
//         return 1111;
//     }
// }

HcclResult _Z18GetRunSideIsDeviceRb(bool &isDeviceSide)
{
    isDeviceSide = false;
    return HcclResult::HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus