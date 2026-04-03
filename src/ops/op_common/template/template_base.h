/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEMPLATE_BASE
#define TEMPLATE_BASE

#include "template_utils.h"
#include "alg_param.h"

namespace ops_hccl {
class TemplateBase {
public:
    explicit TemplateBase() {};

    virtual ~TemplateBase();

    virtual HcclResult KernelRun(const OpParam& param,
                                 const TemplateDataParams& templateDataParams,
                                 TemplateResource& templateResource);
    virtual HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                               AlgResourceRequest& resourceRequest);
};
}
#endif // TEMPLATE_BASE