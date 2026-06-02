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
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/infer_datatype_context.h"
#include "op_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus HcomAllGatherInferShapeV2(gert::InferShapeContext *context)
{
    OP_INFER_SHAPE_START;
 
    // Get RuntimeAttrs
    auto attrs = context->GetAttrs();
    constexpr size_t fusionIndex = 2;
    constexpr size_t fusionIdIndex = 3;
    if (CheckOPAttr(opName, attrs, fusionIndex, fusionIdIndex) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);
 
    constexpr size_t rankIndex = 0;
    int64_t rankSize = *(attrs->GetAttrPointer<int64_t>(rankIndex));
    OP_CHECK((rankSize <= 0),
        CUBE_INNER_ERR_REPORT(opName, "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize), return GRAPH_FAILED);
    // not ShapeFirstDimDefined
    if (inputShape->GetDimNum() > 0 && inputShape->GetDim(0) == ge::UNKNOWN_DIM) {
        *outputShape = *inputShape;
        OP_LOGI(opName, "the op infershape end, shape first dim is unknown.");
        return GRAPH_SUCCESS;
    }
    *outputShape = *inputShape;
    outputShape->SetDim(0, inputShape->GetDim(0) * rankSize);
    
    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}
 
static ge::graphStatus HcomAllGatherInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_INFER_DATATYPE_START;
 
    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
 
    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomAllGather).InferShape(HcomAllGatherInferShapeV2).InferDataType(HcomAllGatherInferDataTypeV2);

}  // namespace ops