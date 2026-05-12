/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_COMMON_H
#define OPS_HCCL_P2P_COMMON_H

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <acl/acl_rt.h>

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t ALG_TAG_LENGTH = TAG_LENGTH + 128;
constexpr uint32_t OP_ALG_LENGTH = 128;
constexpr uint64_t CCU_MAX_RANK_SIZE = 16;

using ChannelHandle = uint64_t;
using ThreadHandle = uint64_t;

// 算子执行模式
enum class OpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

// 通信引擎类型
enum class CommEngine {
    COMM_ENGINE_RESERVED = 0,
    COMM_ENGINE_CPU = 1,
    COMM_ENGINE_AICPU = 2,
    COMM_ENGINE_CPU_TS = 3,
    COMM_ENGINE_AICPU_TS = 4,
    COMM_ENGINE_AIV = 5,
    COMM_ENGINE_CCU = 6,
};

// 设备类型
enum DeviceType {
    DEVICE_TYPE_A2 = 0,
    DEVICE_TYPE_A3 = 1,
    DEVICE_TYPE_A5 = 2,
};

// 算法类型
enum class AlgType {
    ALG_TYPE_MESH_1D = 0,
    ALG_TYPE_NHR = 1,
    ALG_TYPE_RESERVED
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct CcuKernelArgBase {
    // std::vector<ChannelHandle> channels;
    ChannelHandle channels[CCU_MAX_RANK_SIZE];
    uint32_t      channelCount;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel资源组序号，group号不同时，资源复用
    u32 resGroup = 0;
    // kernel名 string？
    char kernelFuncName[64];
    // kernel函数
    void* kernelFunc;
    // KernelArg实例指针
    void *kernelArg;
    // kernel所需channel
    std::vector<HcclChannelDesc> channels;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template<typename T>
    void setKernelArg(std::shared_ptr<T> arg) {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void*>(arg.get());
    }
};

struct KernelResourceRequest {
    std::vector<CcuKernelInfo> ccuKernelInfos;
    std::vector<u32> ccuKernelNum;
};

struct AlgResourceCtxSerializable {
    CommBuffer cclMem;
    uint32_t notifyNumOnMainThread;
    uint32_t slaveThreadNum;
    std::vector<uint32_t> notifyNumPerThread;
    std::vector<ThreadHandle> threads;
    // ccu
    std::vector<u32> ccuKernelNum;
    std::vector<CcuKernelHandle> ccuKernels;

     std::vector<char> Serialize()
    {
        BinaryStream binaryStream;

        binaryStream << cclMem;
        binaryStream << notifyNumOnMainThread;
        binaryStream << slaveThreadNum;
        binaryStream << notifyNumPerThread;
        binaryStream << threads;
        binaryStream << ccuKernelNum;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);

        binaryStream >> cclMem;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> slaveThreadNum;
        binaryStream >> notifyNumPerThread;
        binaryStream >> threads;
        binaryStream >> ccuKernelNum;
        binaryStream >> ccuKernels;
    }

};

// 算子参数
struct OpParam {
    void* hcclComm;
    char tag[TAG_LENGTH];
    char algTag[ALG_TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    aclrtStream stream;
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    OpMode opMode;
    CommEngine engine = CommEngine::COMM_ENGINE_RESERVED;
    AlgType algType;
    char algName[OP_ALG_LENGTH];
    u64 count;
    HcclDataType dataType;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    u64 ctxSize = 0;
    void* resCtx = nullptr;
    u32 myRank = 0;
    u32 rankSize = 0;
    DeviceType devType = DEVICE_TYPE_A5;
};

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

// CCU返回码转换为HCCL返回码
inline HcclResult ConvertCcuToHccl(CcuResult ccuResult) {
    switch (ccuResult) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_MEMORY: return HCCL_E_MEMORY;
        case CCU_E_INTERNAL: return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        case CCU_E_SYSCALL: return HCCL_E_SYSCALL;
        case CCU_E_TIMEOUT: return HCCL_E_TIMEOUT;
        case CCU_E_OPEN_FILE_FAILURE: return HCCL_E_OPEN_FILE_FAILURE;
        case CCU_E_TCP_CONNECT: return HCCL_E_TCP_CONNECT;
        case CCU_E_ROCE_CONNECT: return HCCL_E_ROCE_CONNECT;
        case CCU_E_TCP_TRANSFER: return HCCL_E_TCP_TRANSFER;
        case CCU_E_ROCE_TRANSFER: return HCCL_E_ROCE_TRANSFER;
        case CCU_E_RUNTIME: return HCCL_E_RUNTIME;
        case CCU_E_DRV: return HCCL_E_DRV;
        case CCU_E_PROFILING: return HCCL_E_PROFILING;
        case CCU_E_CCE: return HCCL_E_CCE;
        case CCU_E_NETWORK: return HCCL_E_NETWORK;
        case CCU_E_AGAIN: return HCCL_E_AGAIN;
        case CCU_E_REMOTE: return HCCL_E_REMOTE;
        case CCU_E_SUSPENDING: return HCCL_E_SUSPENDING;
        case CCU_E_OPRETRY_FAIL:
        case CCU_E_OOM:
        case CCU_E_IN_STATUS:
            return HCCL_E_INTERNAL;
        default:
            return HCCL_E_INTERNAL;
    }
}

#endif // OPS_HCCL_P2P_COMMON_H