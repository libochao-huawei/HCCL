/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor -- trans rmt mem to loc mem
 * Author: z00445483
 */

#ifndef HCCL_SIM_TRANS_RMTMEM_TO_LOCMS_EXECUTOR_H
#define HCCL_SIM_TRANS_RMTMEM_TO_LOCMS_EXECUTOR_H

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

class TransRmtMemToLocMemExecutor : public CcuExecutorBase {
public:
    explicit TransRmtMemToLocMemExecutor(int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    TransRmtMemToLocMemExecutor() = default;
    ~TransRmtMemToLocMemExecutor() = default;

    void Parser() override;
    void Run() override;
    std::string Describe() override;

private:
    void Process(CcuResouceManager &ccuResMgr);

private:
    uint16_t locGSAId_{0};
    uint16_t locXnId_{0};
    uint16_t rmtGSAId_{0};
    uint16_t rmtXnId_{0};
    uint16_t lengthXnId_{0};
    uint16_t channelId_{0};
    uint16_t reduceDataType_{0};
    uint16_t reduceOpCode_{0};
    uint16_t clearType_{0};
    uint16_t lengthEn_{0};
    uint16_t reduceEn_{0};
    uint16_t setCKEId_{0};
    uint16_t setCKEMask_{0};
    uint16_t waitCKEId_{0};
    uint16_t waitCKEMask_{0};
    uint64_t transLength_{0};
};

#endif // HCCL_SIM_TRANS_RMTMEM_TO_LOCMS_EXECUTOR_H