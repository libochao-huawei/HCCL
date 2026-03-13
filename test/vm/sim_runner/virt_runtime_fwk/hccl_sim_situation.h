/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim situation header
 */

#ifndef HCCL_SIM_SITUATION_H
#define HCCL_SIM_SITUATION_H

#include <map>
#include <iostream>
#include "fwk_types.h"
#include "hccl/hccl_types.h"

class Situation {
public:
    Situation();

    Situation &SetDataType(HcclDataType dataType);
    Situation &SetReduceOp(HcclReduceOp reduceOp);
    Situation &SetRoot(int root);
    Situation &SetOpType(OpType opType);
    Situation &SetCount(u64 count);
    Situation &SetShmSize(u64 shmSize);
    Situation &SetEnv(const std::string &name, const std::string &value);
    Situation &SetClusterType(DeviceType type, int serverCount, int devCount);
    Situation &addUserEnv(std::map<std::string, std::string> &mp);
    Situation &SetAcceleratorMode(Accelerator mode);
    Situation &SetMc2Flag(bool mc2Flag);
    bool IsAicpuAcceleratorMode();
    bool IsCcuAcceleratorMode();

    template <typename T>
    bool SetDataVec(u64 dataCount, std::vector<T> &data);
    void PrintDataVec(void);
    void InitSituationEnv();

    std::pair<int, int> GetClusterType() const
    {
        return {serverNum,deviceNum};
    }

    inline int GetServerNum()
    {
        return serverNum;
    }

    inline OpType GetOpType()
    {
        return opType;
    }

    inline int GetDeviceNum()
    {
        return deviceNum;
    }

    inline HcclDataType GetDataType()
    {
        return dataType;
    }

    inline HcclReduceOp GetReduceOp()
    {
        return reduceOp;
    }

    inline int GetDstRank()
    {
        return dstRank;
    }

    inline u64 GetCount()
    {
        return count;
    }

    inline u64 GetShmSize()
    {
        return shmSize;
    }

    inline int GetRoot()
    {
        return root;
    }

    inline bool GetStaticAddr()
    {
        return staticAddr;
    }

    inline bool GetStaticShape()
    {
        return staticShape;
    }

    inline DeviceType GetDevType()
    {
        return devType;
    }

    inline int GetRankSize()
    {
        return serverNum * deviceNum;
    }

    inline const std::map<std::string, std::string> &GetEnv()
    {
        return envConfigs;
    }

    inline std::vector<void *> &GetDataVec()
    {
        return dataVec;
    }

private:
    OpType   opType{OpType::ALLREDUCE};
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_RESERVED};
    DeviceType  devType{DeviceType::DEV_TYPE_NOSOC};
    Accelerator accelerator{Accelerator::DEFAULT};
    u64      count{1};
    u64      shmSize{0};
    int      dstRank{1};
    int      root{0};
    int      serverNum{1};
    int      deviceNum{1};
    bool     staticAddr{};
    bool     staticShape{};
    bool     mc2Flag{};

    std::vector<void *> dataVec;  // 数据向量
    std::map<std::string, std::string> envConfigs;
};

template <typename T>
bool Situation::SetDataVec(u64 dataCount, std::vector<T> &data)
{
    if (dataCount != data.size()) {
        std::cout << "datasize not equal to dataCount" << std::endl;
        return false;
    }

    if (this->count != dataCount) {
        dataCount = this->count;
        data.resize(count, (T)(2));
    }

    this->dataVec.reserve(dataCount);
    for (u64 idx = 0; idx < dataCount; idx++) {
        this->dataVec.push_back((void *)&data[idx]);
    }

    return true;
}

#endif