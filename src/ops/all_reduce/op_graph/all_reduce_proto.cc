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

static ge::graphStatus HcomAllReduceInferShapeV2(gert::InferShapeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferShape start.", __func__);

    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);
    uint32_t inputSize = context->GetComputeNodeInputNum();

    OP_LOGD(opName, "[%s] the op  inputSize %u ", __func__, inputSize);
    for (uint32_t index = 0; index < inputSize; index++){
        const auto inputShape = context->GetInputShape(index);
        auto outputShape = context->GetOutputShape(index);
        *outputShape = *inputShape;
    }

    OP_LOGI(opName, "[%s] the op inferShape end.", __func__);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllReduceInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferDataType start.", __func__);

    uint32_t inputSize = context->GetComputeNodeInputNum();

    OP_LOGD(opName, "[%s] the op  inputSize %u ", __func__, inputSize);
    for (uint32_t index = 0; index < inputSize; index++){
        ge::DataType inputType = context->GetInputDataType(index);
        context->SetOutputDataType(index, inputType);
    }

    OP_LOGI(opName, "[%s] the op inferDataType end.", __func__);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomAllReduce).InferShape(HcomAllReduceInferShapeV2).InferDataType(HcomAllReduceInferDataTypeV2);
}  // namespace ops