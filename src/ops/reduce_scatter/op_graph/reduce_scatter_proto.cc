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

static constexpr size_t reduceScatterFusionIndex = 1;
static constexpr size_t reduceScatterFusionIdIndex = 2;
static constexpr size_t rankSizeIndex = 4;

static ge::graphStatus HcomReduceScatterInferShapeV2(gert::InferShapeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferShape start.", __func__);
 
    constexpr int64_t fusionAttrNoFuse = 0;
    constexpr int64_t fusionAttrFuseById = 2;

    constexpr int64_t fusionIdDefaultVal = -1;
    constexpr int64_t fusionIdMinVal = 0;
    constexpr int64_t fusionIdMaxVal = 0x7fffffff;

    // Get RuntimeAttrs
    auto attrs = context->GetAttrs();
    OP_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName, "attrs is null"), return GRAPH_FAILED);

    int64_t fusionAttr = fusionAttrNoFuse;
    int64_t fusionIdAttr = fusionIdDefaultVal;
    
    fusionAttr = *(attrs->GetAttrPointer<int64_t>(reduceScatterFusionIndex));
    fusionIdAttr = *(attrs->GetAttrPointer<int64_t>(reduceScatterFusionIdIndex));

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
 
    int64_t rankSize = *(attrs->GetAttrPointer<int64_t>(rankSizeIndex));
    if (rankSize <= 0){
        OP_LOGE(opName, "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    // not ShapeFirstDimDefined
    if (inputShape->GetDimNum() > 0 && inputShape->GetDim(0) == ge::UNKNOWN_DIM) {
        *outputShape = *inputShape;
        OP_LOGI(opName, "the op infershape end, shape first dim is unknown.");
        return GRAPH_SUCCESS;
    }

    if(inputShape->GetDim(0) % rankSize){
        CUBE_INNER_ERR_REPORT(opName, "input tensor's first dim is illegal, expected: rankSize[%ld] * N "
            "(N is positive integer), actual: %ld.", rankSize, inputShape->GetDim(0));
        return GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    outputShape->SetDim(0, inputShape->GetDim(0) / rankSize);

    OP_LOGI(opName, "[%s] the op inferShape end.", __func__);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomReduceScatterInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferDataType start.", __func__);
 
    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
 
    OP_LOGI(opName, "[%s] the op inferDataType end.", __func__);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomReduceScatter).InferShape(HcomReduceScatterInferShapeV2).InferDataType(HcomReduceScatterInferDataTypeV2);
}  // namespace ops