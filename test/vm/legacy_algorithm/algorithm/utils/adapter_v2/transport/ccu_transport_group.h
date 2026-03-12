/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu transport group header file
 * Create: 2025-02-18
 */

#ifndef HCCL_CCU_TRANS_GROUP_H
#define HCCL_CCU_TRANS_GROUP_H

#include <vector>
#include "ccu_transport.h"
#include "ccu_device_manager.h"

namespace Hccl {

MAKE_ENUM(TransportGrpStatus, INIT, FAIL)

class CcuTransportGroup {
public:
    explicit                CcuTransportGroup(const vector<CcuTransport*> &transports, u32 cntCkeNum);
    CcuTransportGroup() =   delete;     // 禁用默认构造函数
    virtual                ~CcuTransportGroup();
    void                    Destroy();
    TransportGrpStatus      GetGrpStatus() const;
    u32                     GetCntCkeId(u32 index) const;
    bool                    CheckTransports(const vector<CcuTransport*> &transports);
    bool                    CheckTransportCntCke();
    const vector<CcuTransport*> &GetTransports() const;

private:
    vector<CcuTransport*>                   transportsGrp;
    vector<u32>                             cntCkesGroup;
    u32                                     cntCkesGroupDieId{0};
    u32                                     cntCkeNumTransportGroupUse{0};
    vector<CcuResHandle>         cntResHandleTransportGroupUse;
    vector<ResInfo>              ckeInfoTransportGroupUse;
    bool                                    isDestroyed{false};
    TransportGrpStatus                      grpStatus;
    int32_t                                 devLogicId{0};
};

} // namespace Hccl

#endif // HCCL_CCU_TRANS_GROUP_H