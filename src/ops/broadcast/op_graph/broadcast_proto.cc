/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file hcom_ops_v2.cc
 * \brief
 */

#include "ops_proto_hccl.h"
#include "op_log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/infer_datatype_context.h"
#include "op_util.h"

using namespace ge;

namespace ops {

static constexpr size_t fusionIndex = 2;
static constexpr size_t fusionIdIndex = 3;

static ge::graphStatus HcomBroadcastInferShapeV2(gert::InferShapeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferShape start.", __func__);
 
    constexpr int64_t fusionAttrNoFuse = 0;
    constexpr int64_t fusionAttrFuseById = 2;
    constexpr int64_t fusionIdDefaultVal = -1;
    constexpr int64_t fusionIdMinVal = 0;
    constexpr int64_t fusionIdMaxVal = 0x7fffffff;
 
    int64_t fusionAttr = fusionAttrNoFuse;
    int64_t fusionIdAttr = fusionIdDefaultVal;
    // Get RuntimeAttrs
    auto attrs = context->GetAttrs();
    OP_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName, "attrs is null"), return GRAPH_FAILED);
    fusionAttr = *(attrs->GetAttrPointer<int64_t>(fusionIndex));
    fusionIdAttr = *(attrs->GetAttrPointer<int64_t>(fusionIdIndex));
    if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
        OP_LOGE(opName, "Attr fusion [%ld] is not supported. expected: [%ld or %ld]",
                fusionAttr, fusionAttrNoFuse, fusionAttrFuseById);
        return GRAPH_FAILED;
    }
    if (fusionAttr == fusionAttrFuseById) {
        if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
            OP_LOGE(opName, "In fusion [%ld], attr fusion_id [%ld] is not supported, "
                    "expected: [%ld ~ %ld]", fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
            return GRAPH_FAILED;
        }
    }
 
    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);
 
    uint32_t UINT_MAX_VALUE = 0xFFFFFFFF;
    uint32_t inputSize = context->GetComputeNodeInputNum();
    if (inputSize >= UINT_MAX_VALUE) {
        CUBE_INNER_ERR_REPORT(opName, "GetInputSize [%u] is more than %u", inputSize, UINT_MAX_VALUE);
        return GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputSize; i++) {
        const auto inputShape = context->GetInputShape(i);
        OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
        auto outputShape = context->GetOutputShape(i);
        OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);
        *outputShape = *inputShape;
    }
    OP_LOGI(opName, "[%s] the op inferShape end.", __func__);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomBroadcastInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferDataType start.", __func__);

    const unsigned int UINT_MAX_VALUE = 0xFFFFFFFF;
    uint32_t inputSize = context->GetComputeNodeInputNum();
    if (inputSize >= UINT_MAX_VALUE) {
        OP_LOGE(opName, "GetInputSize [%u] is more than %u", inputSize, UINT_MAX_VALUE);
        return GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputSize; i++) {
        ge::DataType inputType = context->GetInputDataType(i);
        context->SetOutputDataType(i, inputType);
    }
 
    OP_LOGI(opName, "[%s] the op inferDataType end.", __func__);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomBroadcast).InferShape(HcomBroadcastInferShapeV2).InferDataType(HcomBroadcastInferDataTypeV2);
}  // namespace ops