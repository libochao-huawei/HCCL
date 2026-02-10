/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STORAGE_MANAGER_H
#define STORAGE_MANAGER_H

#include <vector>
#include <map>
#include <mutex>
#include "task_meta_defs.h"
#include "sim_common.h"
#include "data_slice.h"
#include "binary_data_type_pub.h"


namespace HcclSim {

struct MemBlock {
    uint64_t startAddr;    // 物理起始地址
    uint64_t size;         // 块大小
    BufferType  bufferType;         // 类型
    uint64_t globalOffset; // 该类型总偏移
};

struct CheckerParam {
    HcclCMDType cmdType = static_cast<HcclCMDType>(0);
    uint32_t rankSize = 0;
    HcclDataType dataType = static_cast<HcclDataType>(0);
    uint64_t dataCount = 0;
    
    // 以下是可能缺失的字段，初始化为安全值
    HcclReduceOp reduceType = static_cast<HcclReduceOp>(0);
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t root = 0;

    VDataDesTagInner vDataDes;
    All2AllDataDesTagInner all2AllDataDes;
};

struct RemoteDieInfo {
    uint32_t dstRank;
    uint32_t remoteDieId;
};

using ChannelsPerDie = std::map<uint32_t, RemoteDieInfo>;

class StorageManager {
public:
    static StorageManager& GetInstance() {
        static StorageManager instance;
        return instance;
    }
    // 禁用拷贝构造和赋值
    StorageManager(const StorageManager&) = delete;
    StorageManager& operator=(const StorageManager&) = delete;

    void SetDataId(const std::string& data_id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data_id = data_id;
    }

    std::string GetDataId() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_data_id;
    }

    CheckerParam GetCheckerParam() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_checker_param;
    }

    // 大小太大不缓存
    std::vector<HcclTaskMetaData> GetTaskCollection();

    HcclResult LoadCheckerParam();
    HcclResult LoadMemLayout();
    HcclResult LoadHcclVmSynthesisData();
    HcclResult LoadHcclVmInstrData();
    HcclResult LoadHcclVmTaskMetaData();
    uint64_t GetBlockSize(uint32_t rankId, BufferType bufferType);
    DataSlice GetDataSlice(uint32_t rankId, uint64_t addr, uint64_t size);
    HcclResult GetSlice(uint64_t addr, uint64_t len, DataSlice& dataSlice, uint32_t* rank = nullptr);
    uint32_t GetRankSize() const;

    HcclResult ReadHeader(FILE *fp, FileHeader &header);
    HcclResult ChannelWrite(FILE *fp, const ChannelInfo &chInfo);
    HcclResult ChannelRead(FILE *fp, ChannelInfo &chInfo);

    HcclVmInstrData GetHvmInstrData() const;
    HcclResult Trans2CheckerParam();
    HcclVmTaskMetaData GetHvmTaskMetaData() const;

private:
    StorageManager() = default;

    std::string FindRootPath();
    bool IsDirExists(const std::string& path);

    std::string m_data_id;
    mutable std::mutex m_mutex;

    // RankID -> Type -> StartAddr -> BlockInfo
    std::map<uint32_t, std::map<BufferType, std::map<uint64_t, MemBlock>>> m_mem_layout;
    std::map<RankId, std::map<uint32_t, ChannelsPerDie>> m_allRankChannelInfo;
    CheckerParam m_checker_param;

    HcclVmSynData m_synData;
    HcclVmInstrData m_instrData;
    HcclVmTaskMetaData m_taskMeataData;
};

}

#endif