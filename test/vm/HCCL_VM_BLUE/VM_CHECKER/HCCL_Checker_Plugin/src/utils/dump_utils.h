/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_DUMP_UTILS_H
#define HCCLV1_DUMP_UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include <nlohmann/json.hpp>
#include "hccl_types.h"
#include "sim_task.h"
#include "task_meta_defs.h"
#include "task_def.h"
#include "hccl_vm_log.h"

using json = nlohmann::json;

enum class DumpDataType {
    TASK_META,
    TASK_STUB
};

template<typename T>
std::string ToHexStr(T val) {
    std::stringstream ss;
    ss << "0x" << std::hex << std::uppercase << (static_cast<uint64_t>(val));
    return ss.str();
}

union ResId {
    uint64_t value{UINT64_MAX};
    struct {
        uint64_t podId : 8;
        uint64_t serId : 16;
        uint64_t phyId : 8;
        uint64_t resId : 32;
    } field;

    ResId(uint64_t val) : value(val) {}

    std::string ToString() const {
        std::stringstream ss;
        ss << "(" << static_cast<uint32_t>(field.podId) << "," 
        << static_cast<uint32_t>(field.serId) << "," 
        << static_cast<uint32_t>(field.phyId) << "," 
        << field.resId << ")";
        return ss.str();
    }
};

std::string GetFileName(DumpDataType dumpDataType);
std::string HcclCMDTypeToString(const HcclCMDType &cmd);
std::string HcclDataTypeToString(const HcclDataType &dataType);
json GenBasicInfo(DumpDataType dumpDataType, uint32_t taskCount);
json TaskMetaToJson(const HcclTaskMetaData& s);
void AssignGlobalIndices(HcclSim::TaskNode* head, uint32_t& globalCounter);
json TaskNodePtrToJson(const HcclSim::TaskNodePtr taskNode, u32 &graphCounter);
json DataSliceToJson(const DataSlice &dataSlice);
json LinkProtoStubToJson(const HcclSim::LinkProtoStub &linkProto);
json TaskStubToJson(HcclSim::TaskStub *task, u32 &graphCounter);
std::string HcclReduceOpToString(const HcclReduceOp &reduceOp);
json CcuSingleQueToJson(HcclSim::TaskNodePtr head, u32 rankId, u32 queueIdx, u32 &graphCounter);

HcclResult DumpTaskMetaToFile(const std::vector<HcclTaskMetaData>& taskCollection);
HcclResult DumpTaskStubToFile(HcclSim::TaskNode* headNode);


#endif