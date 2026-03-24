/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context header file
 * Create: 2025-02-18
 */

#ifndef CCU_REP_CTX_H
#define CCU_REP_CTX_H

#include <set>
#include <string>

#include "ccu_rep_base_v1.h"
#include "ccu_rep_block_v1.h"
#include "ccu_kernel_arg.h"

#include "ccu_common.h"

namespace hcomm {
namespace CcuRep {

class CcuRepContext {
public:
    explicit CcuRepContext();
    virtual ~CcuRepContext();

    // 平台层内部使用
    std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock();
    void                                 SetCurrentBlock(std::shared_ptr<CcuRep::CcuRepBlock> repBlock);
    void                                 Append(std::shared_ptr<CcuRep::CcuRepBase> rep);
    const std::vector<std::shared_ptr<CcuRep::CcuRepBase>> &GetRepSequence();
    std::shared_ptr<CcuRep::CcuRepBase> GetRepByInstrId(uint16_t instrId);
    void DumpReprestation();

    void     SetDieId(uint32_t dieId);
    uint32_t GetDieId() const;
    void     SetMissionId(uint32_t missionId);
    uint32_t GetMissionId() const;
    void     SetMissionKey(uint32_t missionKey);
    uint32_t GetMissionKey() const;

protected:
    std::set<std::string> registeredLoop;

private:
    std::shared_ptr<CcuRep::CcuRepBlock> activeBlock{nullptr};
    std::shared_ptr<CcuRep::CcuRepBlock> mainBlock{nullptr};

    uint32_t             dieId{CCU_MAX_IODIE_NUM};
    uint32_t             missionId{UINT32_MAX};
    uint32_t             missionKey{0};
};

}; // namespace CcuRep
}; // namespace hcomm

#endif // _CCU_REP_CTX_H