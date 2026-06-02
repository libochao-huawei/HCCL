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



static ge::graphStatus HcomBroadcastInferShapeV2(gert::InferShapeContext *context)
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
    
    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomBroadcastInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_INFER_DATATYPE_START;

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
 
    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomBroadcast).InferShape(HcomBroadcastInferShapeV2).InferDataType(HcomBroadcastInferDataTypeV2);
}  // namespace ops