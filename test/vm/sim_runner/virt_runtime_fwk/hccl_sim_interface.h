/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim interface header
 */

#ifndef HCCL_SIM_TEST_CASE_H
#define HCCL_SIM_TEST_CASE_H

#include <memory>
#include "hccl_sim_situation.h"
#include "hccl_sim_params.h"
#include "stub_rank_table.h"
#include "CcuResourceManager.h"

class HcclSimInterface {
public:
    explicit HcclSimInterface(Situation &situation, std::string caseName = "")
        : situation(situation), testcaseName(std::move(caseName))
    {

    }
    void Start();
    bool Verify();

private:
    bool aicpuFlag_{false};
    bool ccuFlag_{false};
    bool mc2Flag_{false};
    bool caModelFlag_{0};
    CcuVersion ccuVersionFlag_{CcuVersion::CCU_INVALID};
    Situation situation;
    const std::string testcaseName;
    void InitSituationEnv();
    void SetEnv();
    void UnsetEnv();
    bool InitSimResource();
    void ClearSimResource(DeviceType devType);
    bool CreateCcuInfoDir();
    std::shared_ptr<SimNetworkManager> networkMgr_;
    std::vector<SimParams *> contexts;
};
#endif