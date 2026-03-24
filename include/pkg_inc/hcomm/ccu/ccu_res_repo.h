/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_REPO_H
#define CCU_RES_REPO_H

#include <array>
#include <string>
#include <sstream>
#include <vector>

#include "ccu_common.h"

namespace hcomm {

/*
 * MissionReqType 申请Mission资源的策略类型，当前只按FUSION_MULTIPLE_DIE处理
 * FUSION_MULTIPLE_DIE missionid连续，跨die的missionid相同
 * FUSION_ONE_DIE missionid连续，单die
 * NO_FUSION_ONE_DIE missionid不要求连续，单die
*/
enum class MissionReqType {
    COMM_ENGINE_RESERVED = -1,
    FUSION_MULTIPLE_DIE = 0,
    FUSION_ONE_DIE = 1,
    NO_FUSION_ONE_DIE = 2,
};

class ResInfo {
public:
    ResInfo(): startId(0), num(0){};
    ResInfo(uint32_t startId, uint32_t num) : startId(startId), num(num){};
    uint32_t startId{0};
    uint32_t num{0};

    std::string Describe() const {
        std::ostringstream oss;
        oss << "ResInfo[startId=" << startId << ", num=" << num << "]";
        return oss.str();
    };
};

struct MissionResInfo {
    MissionReqType reqType;
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> mission;
};

struct CcuResRepository {
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> loopEngine{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> blockLoopEngine{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> ms{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> blockMs{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> cke{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> blockCke{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> continuousXn{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> xn{};
    std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM> gsa{};
    MissionResInfo mission{};
};


struct MissionReq {
    MissionReqType reqType{MissionReqType::FUSION_MULTIPLE_DIE};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> req{};
};

struct CcuResReq {
    std::array<uint32_t, CCU_MAX_IODIE_NUM> loopEngineReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> blockLoopEngineReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> msReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> blockMsReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> ckeReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> blockCkeReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> continuousXnReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> xnReq{};
    std::array<uint32_t, CCU_MAX_IODIE_NUM> gsaReq{};
    MissionReq missionReq{};
};

} // namespace hcomm
#endif // CCU_RES_REPO_H