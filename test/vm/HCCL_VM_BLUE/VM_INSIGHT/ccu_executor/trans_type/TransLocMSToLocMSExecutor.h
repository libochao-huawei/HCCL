/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans locms to locms
 * Author: z00445483
 */

#ifndef HCCL_SIM_TRANS_LOCMS_TO_LOCMS_EXECUTOR_H
#define HCCL_SIM_TRANS_LOCMS_TO_LOCMS_EXECUTOR_H

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

class TransLocMSToLocMSExecutor : public CcuExecutorBase {
public:
    explicit TransLocMSToLocMSExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    TransLocMSToLocMSExecutor() = default;
    ~TransLocMSToLocMSExecutor() = default;

    void Parser() override;
    void Run() override;
    void Process(CcuResouceManager &ccuResMgr) override;
    std::string Describe() override;

private:
    uint8_t dstDieId_{0};
    uint8_t srcDieId_{0};
    uint16_t dstMSId_{0};
    uint16_t srcMSId_{0};
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

#endif // HCCL_SIM_TRANS_LOCMS_TO_LOCMS_EXECUTOR_H