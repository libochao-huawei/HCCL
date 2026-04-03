/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "template_base.h"

namespace ops_hccl {

TemplateBase::~TemplateBase()
{
}

HcclResult TemplateBase::KernelRun(const OpParam& param,
                                   const TemplateDataParams& templateDataParams,
                                   TemplateResource& templateResource)
{
    (void)param;
    (void)templateDataParams;
    (void)templateResource;
    HCCL_ERROR("[TemplateBase] Unsupported interface of kernel run!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult TemplateBase::KernelRun(const OpParam& param,
                                   const TemplateDataParams& tempAlgParams,
                                   const TemplateResource& templateResource)
{
    (void)param;
    (void)tempAlgParams;
    (void)templateResource;
    HCCL_ERROR("[TemplateBase] Unsupported interface of kernel run!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult TemplateBase::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                       AlgResourceRequest& resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    HCCL_ERROR("[TemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult TemplateBase::GetRes(AlgResourceRequest& resourceRequest) const
{
    (void)resourceRequest;
    HCCL_ERROR("[TemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

}