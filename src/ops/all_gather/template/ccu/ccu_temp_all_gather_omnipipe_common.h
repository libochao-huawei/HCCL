/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_COMMON_H
#define HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_COMMON_H

#define HCCL_CCU_OMNIPIPE_TEMPLATE_METHODS() \
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, \
                       AlgResourceRequest &resourceRequest) override; \
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams, \
                         TemplateResource &templateResource) override; \
    HcclResult FastLaunch(const OpParam &param, const TemplateFastLaunchCtx &tempFastLaunchCtx) override; \
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override; \
    u64 GetThreadNum() const override; \
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override

#endif // HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_COMMON_H
