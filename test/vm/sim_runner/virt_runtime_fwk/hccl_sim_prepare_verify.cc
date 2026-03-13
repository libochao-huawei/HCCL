/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim verify
 */

#include "hccl_sim_params.h"
#include "hccl_sim_interface.h"
#include "rts_stub.h"
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <functional>
#include "fwk_types.h"
#include "log.h"

const int printNum = 8;

// 为enum class类型自定义哈希函数
namespace std {
template <>
struct hash<HcclDataType> {
    std::size_t operator()(HcclDataType dataType) const
    {
        return std::hash<int>()(static_cast<int>(dataType));
    }
};

template <>
struct hash<OpType> {
    std::size_t operator()(OpType opType) const
    {
        return std::hash<int>()(static_cast<int>(opType));
    }
};
}  // namespace std

// 对各个通信算子进行prepare
bool PrepareSimForBuf(SimParams *params)
{
    try {
        void *sendBuf = nullptr;
        void *recvBuf = nullptr;

        auto dataTypeSize = DataTypeSizeGet(params->situation.GetDataType());
        auto count = params->situation.GetCount();
        auto opType = params->situation.GetOpType();
        u64 sendmemSize = (u64)dataTypeSize * (u64)count;
        u64 recvmemSize = 0;
        auto rankSize = (u64)params->situation.GetRankSize();
        if ((opType == OpType::ALLTOALL || opType == OpType::REDUCESCATTER) && (count % rankSize != 0)) {
            HCCL_ERROR("[Start][PrepareSimForBuf] count must be an integer multiple of rankSize.");
            return false;
        }
       
        switch (opType) {
            case OpType::ALLTOALL:     
                params->sendCount = count / rankSize;
                params->recvCount = count / rankSize;
                recvmemSize = sendmemSize;
                break;
            case OpType::ALLREDUCE:
            case OpType::REDUCE:
            case OpType::BROADCAST:
                recvmemSize = sendmemSize;
                break;
            case OpType::ALLGATHER:
                recvmemSize = sendmemSize * rankSize;
                params->sendCount = count;
                break;
            case OpType::REDUCESCATTER:
                params->recvCount = count / rankSize;
                recvmemSize = sendmemSize / rankSize;
                break;
            default:
                HCCL_ERROR("[Start][PrepareSimForBuf] noly support 6 kind of communicator operator now.");
                return false;
        }
        rtMalloc(&sendBuf, sendmemSize, RT_MEMORY_P2P_HBM, 2);
        rtMalloc(&recvBuf, recvmemSize, RT_MEMORY_P2P_HBM, 2);

        params->recvmemSize = recvmemSize;
        params->sendBuf = sendBuf;
        params->recvBuf = recvBuf;
    } catch (const std::exception &e) {
        HCCL_ERROR("[Start][PrepareSimForBuf] catch some exception:[%s]", e.what());
        return false;
    }

    return true;
}

// 通过模板函数进行扩充Allreduce 算子数据类型的种类
template <typename T>
bool PrepareSimForAllReduceType(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForAllReduceType(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    auto reduceOpType = params->situation.GetReduceOp();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        if (reduceOpType == HcclReduceOp::HCCL_REDUCE_SUM) {
            *exptBuf = value * (T)rankSize;
        } else if (reduceOpType == HcclReduceOp::HCCL_REDUCE_MIN || reduceOpType == HcclReduceOp::HCCL_REDUCE_MAX) {
            *exptBuf = value;
        } else {
            *exptBuf = value;
            for (auto i = 1; i < rankSize; i++) {
                *exptBuf *= value;
            }
        }
        exptBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForAllReduceFp16(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    auto reduceOpType = params->situation.GetReduceOp();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        if (reduceOpType == HcclReduceOp::HCCL_REDUCE_SUM) {
            *exptBuf = value * T(static_cast<float>(rankSize));
        } else if (reduceOpType == HcclReduceOp::HCCL_REDUCE_MIN || reduceOpType == HcclReduceOp::HCCL_REDUCE_MAX) {
            *exptBuf = value;
        }
        exptBuf++;
    }

    return true;
}

template <typename T>
bool PrepareSimForAllReduceFp16(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    auto reduceOpType = params->situation.GetReduceOp();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(SimParams *)>> allreduceSimFuncMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16, [](SimParams *params) { return PrepareSimForAllReduceFp16<FP16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8, [](SimParams *params) { return PrepareSimForAllReduceType<s8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16, [](SimParams *params) { return PrepareSimForAllReduceType<s16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32, [](SimParams *params) { return PrepareSimForAllReduceType<s32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, [](SimParams *params) { return PrepareSimForAllReduceType<u8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, [](SimParams *params) { return PrepareSimForAllReduceType<u16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, [](SimParams *params) { return PrepareSimForAllReduceType<u32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32, [](SimParams *params) { return PrepareSimForAllReduceType<float>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64, [](SimParams *params) { return PrepareSimForAllReduceType<double>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64, [](SimParams *params) { return PrepareSimForAllReduceType<s64>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, [](SimParams *params) { return PrepareSimForAllReduceType<u64>(params); }}
};

bool PrepareSimForAllReduce(SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = allreduceSimFuncMap.find(dataType);
    if (it != allreduceSimFuncMap.end()) {
        return it->second(params);
    }

    HCCL_ERROR("[Start][PrepareSimForAllReduce] not support Allreduce other datatype now.");
    return false;
}

template <typename T>
bool PrepareSimForReduceType(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto root = params->situation.GetRoot();
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto reduceOpType = params->situation.GetReduceOp();

    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForReduceType(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto root = params->situation.GetRoot();
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto reduceOpType = params->situation.GetReduceOp();

    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        if (reduceOpType == HcclReduceOp::HCCL_REDUCE_SUM) {
            if (params->myRank == root) {
                *exptBuf = value * (T)rankSize;
            }
        } else if (reduceOpType == HcclReduceOp::HCCL_REDUCE_MIN || reduceOpType == HcclReduceOp::HCCL_REDUCE_MAX) {
            if (params->myRank == root) {
                *exptBuf = value;
            }
        } else {  // ReduceOp::PROD 乘积
            if (params->myRank == root) {
                *exptBuf = value;
                for (auto i = 1; i < rankSize; i++) {
                    *exptBuf *= value;
                }
            }
        }
        exptBuf++;
    }

    return true;
}

template <typename T>
bool PrepareSimForReduceFp16(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto root = params->situation.GetRoot();
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto reduceOpType = params->situation.GetReduceOp();

    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForReduceFp16(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto root = params->situation.GetRoot();
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto reduceOpType = params->situation.GetReduceOp();

    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        if (reduceOpType == HcclReduceOp::HCCL_REDUCE_SUM) {
            if (params->myRank == root) {
                *exptBuf = value * T(static_cast<float>(rankSize));
            }
        } else if (reduceOpType == HcclReduceOp::HCCL_REDUCE_MIN || reduceOpType == HcclReduceOp::HCCL_REDUCE_MAX) {
            if (params->myRank == root) {
                *exptBuf = value;
            }
        }
        exptBuf++;
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(SimParams *)>> reduceSimFuncMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16, [](SimParams *params) { return PrepareSimForReduceFp16<FP16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8, [](SimParams *params) { return PrepareSimForReduceType<s8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16, [](SimParams *params) { return PrepareSimForReduceType<s16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32, [](SimParams *params) { return PrepareSimForReduceType<s32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, [](SimParams *params) { return PrepareSimForReduceType<u8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, [](SimParams *params) { return PrepareSimForReduceType<u16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, [](SimParams *params) { return PrepareSimForReduceType<u32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32, [](SimParams *params) { return PrepareSimForReduceType<float>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64, [](SimParams *params) { return PrepareSimForReduceType<double>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64, [](SimParams *params) { return PrepareSimForReduceType<s64>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, [](SimParams *params) { return PrepareSimForReduceType<u64>(params); }}
};

bool PrepareSimForReduce(SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = reduceSimFuncMap.find(dataType);
    if (it != reduceSimFuncMap.end()) {
        return it->second(params);
    }

    HCCL_ERROR("[Start][PrepareSimForReduce] not support AllReduce other datatype now.");
    return false;
}

template <typename T>
bool PrepareSimForBroadCastType(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    int root = params->situation.GetRoot();
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    OpType opType = params->situation.GetOpType();

    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        if (params->myRank == root) {
            *sendBuf = value;
            sendBuf++;
        }
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForBroadCastType(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        *exptBuf = value;
        exptBuf++;
    }

    return true;
}

template <typename T>
bool PrepareSimForBroadCastFp16(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    int root = params->situation.GetRoot();
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        if (params->myRank == root) {
            *sendBuf = value;
            sendBuf++;
        }
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForBroadCastFp16(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    int root = params->situation.GetRoot();
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        *exptBuf = value;
        exptBuf++;
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(SimParams *)>> broadcastSimFuncMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16, [](SimParams *params) { return PrepareSimForBroadCastFp16<FP16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8, [](SimParams *params) { return PrepareSimForBroadCastType<s8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16, [](SimParams *params) { return PrepareSimForBroadCastType<s16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32, [](SimParams *params) { return PrepareSimForBroadCastType<s32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, [](SimParams *params) { return PrepareSimForBroadCastType<u8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, [](SimParams *params) { return PrepareSimForBroadCastType<u16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, [](SimParams *params) { return PrepareSimForBroadCastType<u32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32, [](SimParams *params) { return PrepareSimForBroadCastType<float>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64, [](SimParams *params) { return PrepareSimForBroadCastType<double>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64, [](SimParams *params) { return PrepareSimForBroadCastType<s64>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, [](SimParams *params) { return PrepareSimForBroadCastType<u64>(params); }}
};

bool PrepareSimForBroadCast(SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = broadcastSimFuncMap.find(dataType);
    if (it != broadcastSimFuncMap.end()) {
        return it->second(params);
    }

    HCCL_ERROR("[Start][PrepareSimForBroadCast] not support BroadCast other datatype now.");
    return false;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> broadcastExpectBufMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastFp16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCastType<u64>(expectBuf, params); }}
};

bool PrepareExpectBufForBroadCast(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = broadcastExpectBufMap.find(dataType);
    if (it != broadcastExpectBufMap.end()) {
        return it->second(expectBuf, params);
    }

    HCCL_ERROR("[Verify][PrepareExpectBufForBroadCast] not support BroadCast other datatype now.");
    return false;
}

template <typename T>
bool PrepareSimForAllgatherType(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForAllgatherType(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto rankId = 0; rankId < rankSize; rankId++) {
        for (auto idx = 0; idx < count; idx++) {
            T value = *(T *)InputData[idx];
            *exptBuf = value;
            exptBuf++;
        }
    }

    return true;
}

template <typename T>
bool PrepareSimForAllgatherFp16(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForAllgatherFp16(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    for (auto rankId = 0; rankId < rankSize; rankId++) {
        for (auto idx = 0; idx < count; idx++) {
            T value(*(T *)InputData[idx]);
            *exptBuf = value;
            exptBuf++;
        }
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(SimParams *)>> allgatherSimFuncMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16, [](SimParams *params) { return PrepareSimForAllgatherFp16<FP16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8, [](SimParams *params) { return PrepareSimForAllgatherType<s8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16, [](SimParams *params) { return PrepareSimForAllgatherType<s16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32, [](SimParams *params) { return PrepareSimForAllgatherType<s32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, [](SimParams *params) { return PrepareSimForAllgatherType<u8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, [](SimParams *params) { return PrepareSimForAllgatherType<u16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, [](SimParams *params) { return PrepareSimForAllgatherType<u32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32, [](SimParams *params) { return PrepareSimForAllgatherType<float>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64, [](SimParams *params) { return PrepareSimForAllgatherType<double>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64, [](SimParams *params) { return PrepareSimForAllgatherType<s64>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, [](SimParams *params) { return PrepareSimForAllgatherType<u64>(params); }}
};

bool PrepareSimForAllgather(SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = allgatherSimFuncMap.find(dataType);
    if (it != allgatherSimFuncMap.end()) {
        return it->second(params);
    }

    HCCL_ERROR("[Start][PrepareSimForAllgather] not support Allgather other datatype now.");
    return false;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> allgatherExpectBufMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherFp16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgatherType<u64>(expectBuf, params); }}
};

bool PrepareExpectBufForAllgather(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = allgatherExpectBufMap.find(dataType);
    if (it != allgatherExpectBufMap.end()) {
        return it->second(expectBuf, params);
    }

    HCCL_ERROR("[Verify][PrepareExpectBufForAllgather] not support Allgather other datatype now.");
    return false;
}

template <typename T>
bool PrepareSimForAlltoAllType(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto smallcount = count / rankSize;
    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForAlltoAllType(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto smallcount = count / rankSize;
    for (auto idx = 0; idx < count; idx = idx + smallcount) {
        for (auto idy = 0; idy < smallcount; idy++) {
            *exptBuf = *(T *)InputData[(params->myRank) * smallcount + idy];
            exptBuf++;
        }
    }

    return true;
}

template <typename T>
bool PrepareSimForAlltoAllFp16(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto smallcount = count / rankSize;
    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForAlltoAllFp16(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto smallcount = count / rankSize;
    for (auto idx = 0; idx < count; idx = idx + smallcount) {
        for (auto idy = 0; idy < smallcount; idy++) {
            *exptBuf = *(T *)InputData[(params->myRank) * smallcount + idy];
            exptBuf++;
        }
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(SimParams *)>> alltoallSimFuncMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16, [](SimParams *params) { return PrepareSimForAlltoAllFp16<FP16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8, [](SimParams *params) { return PrepareSimForAlltoAllType<s8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16, [](SimParams *params) { return PrepareSimForAlltoAllType<s16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32, [](SimParams *params) { return PrepareSimForAlltoAllType<s32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, [](SimParams *params) { return PrepareSimForAlltoAllType<u8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, [](SimParams *params) { return PrepareSimForAlltoAllType<u16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, [](SimParams *params) { return PrepareSimForAlltoAllType<u32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32, [](SimParams *params) { return PrepareSimForAlltoAllType<float>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64, [](SimParams *params) { return PrepareSimForAlltoAllType<double>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64, [](SimParams *params) { return PrepareSimForAlltoAllType<s64>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, [](SimParams *params) { return PrepareSimForAlltoAllType<u64>(params); }}
};

bool PrepareSimForAlltoAll(SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = alltoallSimFuncMap.find(dataType);
    if (it != alltoallSimFuncMap.end()) {
        return it->second(params);
    }

    HCCL_ERROR("[Start][PrepareSimForAlltoAll] not support AlltoAll other datatype now.");
    return false;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> alltoallExpectBufMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllFp16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAllType<u64>(expectBuf, params); }}
};

bool PrepareExpectBufForAlltoAll(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = alltoallExpectBufMap.find(dataType);
    if (it != alltoallExpectBufMap.end()) {
        return it->second(expectBuf, params);
    }

    HCCL_ERROR("[Verify][PrepareExpectBufForAlltoAll] not support AlltoAll other datatype now.");
    return false;
}

template <typename T>
bool PrepareSimForReduceScatterType(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto ReduceOpType = params->situation.GetReduceOp();
    int expBufSize = count / rankSize;

    for (auto idx = 0; idx < count; idx++) {
        T value = *(T *)InputData[idx];
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForReduceScatterType(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    int expBufSize = count / rankSize;

    for (auto idx = 0; idx < count; idx++) {
        if (params->myRank != idx) {
            continue;
        }

        T value = *(T *)InputData[idx];
        if (params->situation.GetReduceOp() == HcclReduceOp::HCCL_REDUCE_SUM) {
            int val = idx * expBufSize;
            for (int expBufId = 0; expBufId < expBufSize; expBufId++) {
                T value = *(T *)InputData[val];
                *exptBuf = value * (T)rankSize;
                val++;
                exptBuf++;
            }
        } else if (params->situation.GetReduceOp() == HcclReduceOp::HCCL_REDUCE_MIN ||
                   params->situation.GetReduceOp() == HcclReduceOp::HCCL_REDUCE_MAX) {

            int val = idx * expBufSize;
            for (int expBufId = 0; expBufId < expBufSize; expBufId++) {
                T value = *(T *)InputData[val];
                *exptBuf = value;
                exptBuf++;
                val++;
            }
        } else {  // ReduceOp::PROD 乘积
            int val = idx * expBufSize;
            for (int expBufId = 0; expBufId < expBufSize; expBufId++) {
                T value = *(T *)InputData[val];
                *exptBuf = value;
                for (auto i = 1; i < rankSize; i++) {
                    *exptBuf *= value;
                }
                exptBuf++;
                val++;
            }
        }
    }

    return true;
}

template <typename T>
bool PrepareSimForReduceScatterFp16(SimParams *params)
{
    auto *sendBuf = (T *)params->sendBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    auto ReduceOpType = params->situation.GetReduceOp();
    int expBufSize = count / rankSize;

    for (auto idx = 0; idx < count; idx++) {
        T value(*(T *)InputData[idx]);
        *sendBuf = value;
        sendBuf++;
    }

    return true;
}

template <typename T>
bool PrepareExpectBufForReduceScatterFp16(void *expectBuf, SimParams *params)
{
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    std::vector<void *> &InputData = params->situation.GetDataVec();
    auto rankSize = params->situation.GetRankSize();
    int expBufSize = count / rankSize;

    for (auto idx = 0; idx < count; idx++) {
        if (params->myRank != idx) {
            continue;
        }

        T value(*(T *)InputData[idx]);
        if (params->situation.GetReduceOp() == HcclReduceOp::HCCL_REDUCE_SUM) {
            int val = idx * expBufSize;
            for (int expBufId = 0; expBufId < expBufSize; expBufId++) {
                T value(*(T *)InputData[val]);
                *exptBuf = value * T(static_cast<float>(rankSize));
                val++;
                exptBuf++;
            }
        } else if (params->situation.GetReduceOp() == HcclReduceOp::HCCL_REDUCE_MIN ||
                   params->situation.GetReduceOp() == HcclReduceOp::HCCL_REDUCE_MAX) {
            int val = idx * expBufSize;
            for (int expBufId = 0; expBufId < expBufSize; expBufId++) {
                T value(*(T *)InputData[val]);
                *exptBuf = value;
                exptBuf++;
                val++;
            }
        }
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(SimParams *)>> reducescatterSimFuncMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16, [](SimParams *params) { return PrepareSimForReduceScatterFp16<FP16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8, [](SimParams *params) { return PrepareSimForReduceScatterType<s8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16, [](SimParams *params) { return PrepareSimForReduceScatterType<s16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32, [](SimParams *params) { return PrepareSimForReduceScatterType<s32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, [](SimParams *params) { return PrepareSimForReduceScatterType<u8>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, [](SimParams *params) { return PrepareSimForReduceScatterType<u16>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, [](SimParams *params) { return PrepareSimForReduceScatterType<u32>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32, [](SimParams *params) { return PrepareSimForReduceScatterType<float>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64, [](SimParams *params) { return PrepareSimForReduceScatterType<double>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64, [](SimParams *params) { return PrepareSimForReduceScatterType<s64>(params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, [](SimParams *params) { return PrepareSimForReduceScatterType<u64>(params); }}
};

bool PrepareSimForReduceScatter(SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = reducescatterSimFuncMap.find(dataType);
    if (it != reducescatterSimFuncMap.end()) {
        return it->second(params);
    }

    HCCL_ERROR("[Start][PrepareSimForReduceScatter] not support ReduceScatter other datatype now.");
    return false;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> rdscatterExpectBufMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterFp16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatterType<u64>(expectBuf, params); }}
};

bool PrepareExpectBufForReduceScatter(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = rdscatterExpectBufMap.find(dataType);
    if (it != rdscatterExpectBufMap.end()) {
        return it->second(expectBuf, params);
    }

    HCCL_ERROR("[Verify][PrepareExpectBufForReduceScatter] not support ReduceScatter other datatype now.");
    return false;
}

template <typename T>
bool areFloatsEqual(T a, T b, T epsilon)
{
    auto result = fabs(a - b);
    return fabs(a - b) <= epsilon;
}

template <typename T>
bool areIntsEqual(T a, T b)
{
    return a == b;
}

// 通过模板函数对AllReduce算子进行不同数据类型验证
template <typename T>
bool VerifyReduceExtType(void *expectBuf, SimParams *params)
{
    auto *recvBuf = (T *)params->recvBuf;
    auto *exptBuf = (T *)expectBuf;

    auto count = params->situation.GetCount();
    auto resultFlag = true;
    auto CompareResult = true;
    for (auto idx = 0; idx < count; idx++) {
        T f1 = *recvBuf;
        T f2 = *exptBuf;
        // 可能有超大数据量场景，此处不做全量打印
        // if (idx < printNum) {
        //     std::cout << "rank: " << params->myRank<<", dataCount="<<count << ", " << static_cast<double>(f1) << "------"
        //               << static_cast<double>(f2) << std::endl;
        // }

        switch (params->situation.GetDataType()) {
            case HcclDataType::HCCL_DATA_TYPE_FP16:
            case HcclDataType::HCCL_DATA_TYPE_FP32:
            case HcclDataType::HCCL_DATA_TYPE_FP64:
                CompareResult = areFloatsEqual<T>(f1, f2, 1e-5);  // 精度这里定义成宏
                break;
            default:
                CompareResult = areIntsEqual<T>(f1, f2);
        }
        if (!CompareResult) {
            resultFlag = false;
        }
        recvBuf++;
        exptBuf++;
    }
    return resultFlag;
}

template <typename T>
bool VerifyReduceExtFP16(void *expectBuf, SimParams *params)
{
    auto *recvBuf = (T *)params->recvBuf;
    auto *exptBuf = (T *)expectBuf;
    auto count = params->situation.GetCount();
    for (auto idx = 0; idx < count; idx++) {
        FP16 f1(*recvBuf);
        FP16 f2(*exptBuf);
        // if (idx < printNum) {
        //     std::cout << "rank: " << params->myRank << ", " << f1 << "------" << f2 << std::endl;
        // }

        if (std::fabs(f1.to_float() - f2.to_float()) > 1e-3f) {
            return false;
        }
        recvBuf++;
        exptBuf++;
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> reduceVerifyMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtFP16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return VerifyReduceExtType<u64>(expectBuf, params); }}
};

bool VerifyReduceExt(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = reduceVerifyMap.find(dataType);
    if (it != reduceVerifyMap.end()) {
        return it->second(expectBuf, params);
    }

    return false;
}

bool VerifyBroadCast(void *expectBuf, SimParams *params)
{
    auto dataTypeSize = DataTypeSizeGet(params->situation.GetDataType());
    auto count = params->situation.GetCount();
    u64 recvMemSize = (u64)dataTypeSize * (u64)count;
    if (memcmp(params->sendBuf, expectBuf, recvMemSize) == 0) {
        return true;
    }
    return false;
}

bool VerifyAllgather(void *expectBuf, SimParams *params)
{
    auto dataTypeSize = DataTypeSizeGet(params->situation.GetDataType());
    auto count = params->situation.GetCount();
    u64 recvMemSize = (u64)dataTypeSize * (u64)count * (u64)params->situation.GetRankSize();

    if (memcmp(params->recvBuf, expectBuf, recvMemSize) == 0) {
        return true;
    }
    return false;
}

// 大数据量情况下，分段打印错误数据信息
void DumpData(void *expectBuf, SimParams *params)
{
    auto count = params->situation.GetCount();
    auto recfBuf = (int*)(params->recvBuf);
    auto expBuf = (int*)(expectBuf);
    uint32_t start = 0;
    uint32_t end = 0;
    int startValue0 = 0;
    int startValue1 = 0;
    int endValue0 = 0;
    int endValue1 = 0;
    int flag = 0;
    for (u32 i = 0; i < count; i++) {
        if ((flag != 1) && (*recfBuf != *expBuf)) {
            start = i;
            startValue0 = *recfBuf;
            startValue1 = *expBuf;
            flag = 1;
        }
        if (flag == 1 && *recfBuf == *expBuf) {
            flag = 0;
            end = i-1;
            endValue0 = *(recfBuf - 1);
            endValue1 = *(expBuf - 1);
            std::cout << "[VerifyAlltoAll] not equal: rank" << params->myRank << ": [" << start << "-" << end << "]"
                      << startValue0 << " -- " << startValue1 << ", " << endValue0 << " -- " << endValue1 << std::endl;
        }
        recfBuf++;
        expBuf++;
    }
}

bool VerifyAlltoAll(void *expectBuf, SimParams *params)
{
    auto dataTypeSize = DataTypeSizeGet(params->situation.GetDataType());
    auto count = params->situation.GetCount();
    u64 recvMemSize = (u64)dataTypeSize * (u64)count;

    if (memcmp(params->recvBuf, expectBuf, recvMemSize) == 0) {
        return true;
    }
    return false;
}

// 小数据量情况下，打印错误数据信息
template <typename T>
void DumpData(void *expectBuf, SimParams *params)
{
    auto *recvBuf = (T *)params->recvBuf;
    auto *exptBuf = (T *)expectBuf;
    auto f1Ptr = recvBuf;
    auto f2Ptr = exptBuf;
    auto count = params->situation.GetCount();
    auto rankSize = params->situation.GetRankSize();
    auto expBufSize = count / rankSize;
    for (auto idx = 0; idx < expBufSize; idx++) {
        T f1 = *f1Ptr;
        T f2 = *f2Ptr;
        if (idx < printNum) {
            std::cout << "rankId: " << params->myRank << ", " << static_cast<T>(f1) << "------"
                      << static_cast<T>(f2) << std::endl;
        }
        f1Ptr++;
        f2Ptr++;
    }
}

template <typename T>
bool VerifyReduceScatterType(void *expectBuf, SimParams *params)
{
    auto *recvBuf = (T *)params->recvBuf;
    auto *exptBuf = (T *)expectBuf;
    auto CompareResult = true;
    auto count = params->situation.GetCount();
    auto rankSize = params->situation.GetRankSize();
    auto expBufSize = count / rankSize;

    for (auto idx = 0; idx < expBufSize; idx++) {
        T f1 = *recvBuf;
        T f2 = *exptBuf;

        switch (params->situation.GetDataType()) {
            case HcclDataType::HCCL_DATA_TYPE_FP16:
            case HcclDataType::HCCL_DATA_TYPE_FP32:
            case HcclDataType::HCCL_DATA_TYPE_FP64:
                CompareResult = areFloatsEqual<T>(f1, f2, 1e-5);
                break;
            default:
                CompareResult = areIntsEqual<T>(f1, f2);
        }

        if (!CompareResult) {
            return false;
        }
        recvBuf++;
        exptBuf++;
    }
    return true;
}

template <typename T>
bool VerifyReduceScatterFP16(void *expectBuf, SimParams *params)
{
    auto *recvBuf = (T *)params->recvBuf;
    auto *exptBuf = (T *)expectBuf;
    auto CompareResult = true;
    auto count = params->situation.GetCount();
    auto rankSize = params->situation.GetRankSize();
    auto expBufSize = count / rankSize;

    for (auto idx = 0; idx < expBufSize; idx++) {
        FP16 f1(*recvBuf);
        FP16 f2(*exptBuf);
        // if (idx < printNum) {
        //     std::cout << "rank: " << params->myRank << ", " << f1 << "------" << f2 << std::endl;
        // }

        if (std::fabs(f1.to_float() - f2.to_float()) > 1e-3f) {
            return false;
        }
        recvBuf++;
        exptBuf++;
    }

    return true;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> rdscatterVerifyMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterFP16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return VerifyReduceScatterType<u64>(expectBuf, params); }}
};

bool VerifyReduceScatter(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = rdscatterVerifyMap.find(dataType);
    if (it != rdscatterVerifyMap.end()) {
        return it->second(expectBuf, params);
    }

    return false;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> allreduceExpectBufMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceFp16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduceType<u64>(expectBuf, params); }}
};

bool PrepareExpectBufForAllReduce(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = allreduceExpectBufMap.find(dataType);
    if (it != allreduceExpectBufMap.end()) {
        return it->second(expectBuf, params);
    }

    HCCL_ERROR("[Verify][PrepareExpectBufForAllReduce] not support Allreduce other datatype now.");
    return false;
}

const std::unordered_map<HcclDataType, std::function<bool(void *expectBuf, SimParams *)>> reduceExpectBufMap = {
    {HcclDataType::HCCL_DATA_TYPE_FP16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceFp16<FP16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<s8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<s16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<s32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT8,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<u8>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT16,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<u16>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<u32>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP32,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<float>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_FP64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<double>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_INT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<s64>(expectBuf, params); }},
    {HcclDataType::HCCL_DATA_TYPE_UINT64,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceType<u64>(expectBuf, params); }}
};

bool PrepareExpectBufForReduce(void *expectBuf, SimParams *params)
{
    auto dataType = params->situation.GetDataType();
    auto it = reduceExpectBufMap.find(dataType);
    if (it != reduceExpectBufMap.end()) {
        return it->second(expectBuf, params);
    }

    HCCL_ERROR("[Verify][PrepareExpectBufForReduce] not support Reduce other datatype now.");
    return false;
}

const std::unordered_map<OpType, std::function<bool(void *expectBuf, SimParams *)>> prepareExpectFuncMap = {
    {OpType::ALLREDUCE,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllReduce(expectBuf, params); }},
    {OpType::REDUCE,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduce(expectBuf, params); }},
    {OpType::BROADCAST,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForBroadCast(expectBuf, params); }},
    {OpType::ALLGATHER,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAllgather(expectBuf, params); }},
    {OpType::ALLTOALL,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForAlltoAll(expectBuf, params); }},
    {OpType::REDUCESCATTER,
        [](void *expectBuf, SimParams *params) { return PrepareExpectBufForReduceScatter(expectBuf, params); }}
};

bool PrepareExpectBuf(void **expectBuf, SimParams *params)
{
    try {
        rtHostMalloc(expectBuf, params->recvmemSize, RT_MEMORY_P2P_HBM, 2);
        OpType opType = params->situation.GetOpType();
        auto it = prepareExpectFuncMap.find(opType);
        if (it != prepareExpectFuncMap.end()) {
            return it->second(*expectBuf, params);
        }

        HCCL_ERROR("[Verify][PrepareExpectBuf] not support the opType:[%s].", opType.Describe().c_str());
        return false;
    } catch (const std::exception &e) {
        HCCL_ERROR("[Verify][PrepareExpectBuf] catch some exception:[%s]", e.what());
        return false;
    }

    return false;
}

const std::unordered_map<OpType, std::function<bool(void *expectBuf, SimParams *)>> verifySimFuncMap = {
    {OpType::ALLREDUCE, [](void *expectBuf, SimParams *params) { return VerifyReduceExt(expectBuf, params); }},
    {OpType::REDUCE, [](void *expectBuf, SimParams *params) { return VerifyReduceExt(expectBuf, params); }},
    {OpType::BROADCAST, [](void *expectBuf, SimParams *params) { return VerifyBroadCast(expectBuf, params); }},
    {OpType::ALLGATHER, [](void *expectBuf, SimParams *params) { return VerifyAllgather(expectBuf, params); }},
    {OpType::ALLTOALL, [](void *expectBuf, SimParams *params) { return VerifyAlltoAll(expectBuf, params); }},
    {OpType::REDUCESCATTER, [](void *expectBuf, SimParams *params) { return VerifyReduceScatter(expectBuf, params); }}
};

bool VerifySimResult(SimParams *params)
{
    // 准备用于验证结果的expbuf数据
    void *expectBuf = nullptr;
    if (!PrepareExpectBuf(&expectBuf, params)) {
        return false;
    }

    bool ret = false;
    auto opType = params->situation.GetOpType();
    auto it = verifySimFuncMap.find(opType);
    if (it != verifySimFuncMap.end()) {
        ret = it->second(expectBuf, params);
    } else {
        ret = false;
    }

    if (expectBuf != nullptr) {
        rtHostFree(expectBuf);
    }
    return ret;
}

const std::unordered_map<OpType, std::function<bool(SimParams *)>> prepareSimFuncMap = {
    {OpType::ALLREDUCE, [](SimParams *params) { return PrepareSimForAllReduce(params); }},
    {OpType::REDUCE, [](SimParams *params) { return PrepareSimForReduce(params); }},
    {OpType::BROADCAST, [](SimParams *params) { return PrepareSimForBroadCast(params); }},
    {OpType::ALLGATHER, [](SimParams *params) { return PrepareSimForAllgather(params); }},
    {OpType::ALLTOALL, [](SimParams *params) { return PrepareSimForAlltoAll(params); }},
    {OpType::REDUCESCATTER, [](SimParams *params) { return PrepareSimForReduceScatter(params); }}
};

bool PrepareSimParams(SimParams * params)
{
    if (!PrepareSimForBuf(params)) {
        return false;
    }

    OpType opType = params->situation.GetOpType();
    auto it = prepareSimFuncMap.find(opType);
    if (it != prepareSimFuncMap.end()) {
        return it->second(params);
    }

    HCCL_ERROR("[Start][PrepareSimParams] not support the opType:[%s].", opType.Describe().c_str());
    return false;
}