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

static ge::graphStatus HcomReduceInferShapeV2(gert::InferShapeContext *context)
{
    OP_INFER_SHAPE_START;

    uint32_t inputSize = context->GetComputeNodeInputNum();
    OP_LOGD(opName, "[%s] the op  inputSize %u ", __func__, inputSize);

    for (uint32_t index = 0; index < inputSize; index++) {
        const auto inputShape = context->GetInputShape(index);
        auto outputShape = context->GetOutputShape(index);
        *outputShape = *inputShape;
    }

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomReduceInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_INFER_DATATYPE_START;

    uint32_t inputSize = context->GetComputeNodeInputNum();

    OP_LOGD(opName, "[%s] the op  inputSize %u ", __func__, inputSize);
    for (uint32_t index = 0; index < inputSize; index++) {
        ge::DataType inputType = context->GetInputDataType(index);
        context->SetOutputDataType(index, inputType);
    }

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomReduce).InferShape(HcomReduceInferShapeV2).InferDataType(HcomReduceInferDataTypeV2);
}  // namespace ops