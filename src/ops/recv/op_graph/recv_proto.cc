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

static ge::graphStatus HcomReceiveInferShapeV2(gert::InferShapeContext *context)
{
    OP_INFER_SHAPE_START;

    auto attrs = context->GetAttrs();
    OP_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName, "attrs is null"), return GRAPH_FAILED);

    auto shapeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    OP_CHECK(shapeAttr == nullptr, CUBE_INNER_ERR_REPORT(opName, "shapeAttr is null"), return GRAPH_FAILED);

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    auto sizesArray = reinterpret_cast<const int64_t*>(shapeAttr->GetData());
    outputShape->SetDimNum(shapeAttr->GetSize());

    for (size_t i = 0; i < shapeAttr->GetSize(); ++i) {
        if (sizesArray[i] <= 0) {
            OP_LOGE(opName, "value of sizes[%ld] must greater than 0, but got %lu", i, sizesArray[i]);
            return GRAPH_FAILED;
        }
        outputShape->SetDim(i, sizesArray[i]);
    }

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomReceiveInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomReceive).InferShape(HcomReceiveInferShapeV2).InferDataType(HcomReceiveInferDataTypeV2);
}  // namespace ops