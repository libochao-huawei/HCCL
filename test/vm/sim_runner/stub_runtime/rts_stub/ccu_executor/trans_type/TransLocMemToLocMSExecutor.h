/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans loc mem to loc ms
 * Author: z00445483
 */

#ifndef HCCL_SIM_TRANS_LOCMEM_TO_LOCMS_EXECUTOR_H
#define HCCL_SIM_TRANS_LOCMEM_TO_LOCMS_EXECUTOR_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "rts_stub.h"
#include "FakeStreamMgr.h"
#include "ccuMicrocodeV1.h"
#include "ccu_task_param.h"
#include "CcuExecutorBase.h"
#include "CcuResourceManager.h"

class TransLocMemToLocMSExecutor : public CcuExecutorBase {
public:
    explicit TransLocMemToLocMSExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    TransLocMemToLocMSExecutor() = default;
    ~TransLocMemToLocMSExecutor() = default;

    void Parser() override;
    void Run() override;
    void Process(CcuResouceManager &ccuResMgr) override;
    std::string Describe() override;

private:
    

private:
    uint8_t locDieId_{0};
    uint16_t locGSAId_{0};
    uint16_t locXnId_{0};
    uint16_t locMSId_{0};
    uint16_t lengthXnId_{0};
    uint16_t channelId_{0};
    uint16_t clearType_{0};
    uint16_t lengthEn_{0};
    uint16_t setCKEId_{0};
    uint16_t setCKEMask_{0};
    uint16_t waitCKEId_{0};
    uint16_t waitCKEMask_{0};
    uint16_t transLength_{0};
};

#endif // HCCL_SIM_TRANS_LOCMEM_TO_LOCMS_EXECUTOR_H