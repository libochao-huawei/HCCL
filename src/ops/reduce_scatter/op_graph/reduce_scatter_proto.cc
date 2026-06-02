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

static ge::graphStatus HcomReduceScatterInferShapeV2(gert::InferShapeContext *context)
{
    OP_INFER_SHAPE_START;
 
    // Get RuntimeAttrs
    auto attrs = context->GetAttrs();
    constexpr size_t reduceScatterFusionIndex = 1;
    constexpr size_t reduceScatterFusionIdIndex = 2;
    if (CheckOPAttr(attrs, reduceScatterFusionIndex, reduceScatterFusionIdIndex) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }
    
    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);
 
     constexpr size_t rankSizeIndex = 4;
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

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomReduceScatterInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_INFER_DATATYPE_START;
 
    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
 
    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomReduceScatter).InferShape(HcomReduceScatterInferShapeV2).InferDataType(HcomReduceScatterInferDataTypeV2);
}  // namespace ops