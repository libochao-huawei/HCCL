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

static constexpr size_t fusionIndex = 2;
static constexpr size_t fusionIdIndex = 3;
static constexpr size_t rankIndex = 0;

static bool HcomIsConstData(gert::InferShapeContext *context, const gert::Tensor* shape_tensor){
    if(shape_tensor == nullptr){
        OP_LOGE(context->GetNodeName(), "[%s] the op shape tensor is null.", __func__);
        return false;
    }
    return IsConstTensor(shape_tensor);
}

static void HcomGetConstValue(gert::InferShapeContext *context, const gert::Tensor* const_tensor, 
                              const DataType& dtype, std::vector<int64_t>& const_data) {
    if (dtype == ge::DT_INT64){
        const int64_t* const_data_ptr = const_tensor->GetData<int64_t>();
        size_t size = const_tensor->GetShapeSize();
        OP_LOGD(context->GetNodeName(), "size : %zu",size);
        for (size_t i = 0; i < size; ++i) {
            const_data.push_back(*(const_data_ptr + i));
            OP_LOGD(context->GetNodeName(), "[%s] const data int64  %ld", __func__, (int64_t)(*(const_data_ptr + i)));
        }
    } else if (dtype == ge::DT_INT32) {
        const int32_t* const_data_ptr = const_tensor->GetData<int32_t>();
        size_t size = const_tensor->GetShapeSize();
        for (size_t i = 0; i < size; ++i) {
            const_data.push_back(*(const_data_ptr + i));
            OP_LOGD(context->GetNodeName(), "[%s] const data int32  %d", __func__, (int32_t)(*(const_data_ptr + i)));
        }
    }
    return;
}

void HcomGetConstValue(gert::InferShapeContext *context, const gert::Tensor* const_tensor, 
                       const DataType& dtype, std::vector<uint64_t>& const_data) {
    if (dtype == ge::DT_UINT64){
        const uint64_t* const_data_ptr = const_tensor->GetData<uint64_t>();
        size_t size = const_tensor->GetShapeSize();
        for (size_t i = 0; i < size; ++i) {
            const_data.push_back(*(const_data_ptr + i));
            OP_LOGD(context->GetNodeName(), "[%s] const data int64  %lu", __func__, (uint64_t)(*(const_data_ptr + i)));
        }
    } 
    return;
}

static ge::graphStatus HcomAllToAllVInferShapeV2(gert::InferShapeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferShape start.", __func__);

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    const gert::Tensor* recvDispTensor = context->GetInputTensor(4);
    const gert::Tensor* recvCountsTensor = context->GetInputTensor(3);

    if (!HcomIsConstData(context, recvDispTensor) || !HcomIsConstData(context, recvCountsTensor)){
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        OP_LOGI(opName, "[%s] the op inferShape unknown.", __func__);
        return GRAPH_SUCCESS;
    }

    auto recvDispDtype = context->GetInputTensor(4);
    vector<int64_t> recvDisp;
    HcomGetConstValue(context, recvDispTensor, recvDispDtype->GetDataType(), recvDisp);

    auto recvCountsDtype = context->GetInputTensor(3);
    vector<int64_t> recvCounts;
    HcomGetConstValue(context, recvCountsTensor, recvCountsDtype->GetDataType(), recvCounts);

    if (recvDisp.size() != recvCounts.size()) {
        OP_LOGE(opName, "recvDisp size[%zu] and recvCounts size[%zu] are different.",
            recvDisp.size(), recvCounts.size());
        return GRAPH_FAILED;
    }

    int64_t recvShape = -1;
    for (size_t i = 0; i < recvDisp.size(); i++) {
        int64_t tempSum = recvDisp[i] + recvCounts[i];
        if (recvShape < tempSum) {
            recvShape = tempSum;
        }
    }

    outputShape->SetDimNum(1);
    outputShape->SetDim(0, recvShape);

    OP_LOGI(opName, "[%s] the op inferShape end.", __func__);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllVInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferDataType start.", __func__);

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_LOGI(opName, "[%s] the op inferDataType end.", __func__);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllVCInferShapeV2(gert::InferShapeContext *context)
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

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    const gert::Tensor* sendCountMatrixTensor = context->GetInputTensor(1);

    if (!HcomIsConstData(context, sendCountMatrixTensor)){
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        OP_LOGI(opName, "[%s] the op inferShape unknown.", __func__);
        return GRAPH_SUCCESS;
    }

    vector<int64_t> sendCountMatrix;
    auto sendCountDtype = context->GetInputTensor(1);
    HcomGetConstValue(context, sendCountMatrixTensor, sendCountDtype->GetDataType(), sendCountMatrix);

    for (size_t i = 0;i < sendCountMatrix.size();++i){
        OP_LOGD(opName, "[%s] sendCountMatrix : %zu : %ld ", __func__, i, sendCountMatrix[i]);
    }

    int64_t rank = *(attrs->GetAttrPointer<int64_t>(rankIndex));
    int64_t rankSize = static_cast<int64_t>(sqrt(sendCountMatrix.size()));
    if (rankSize <= 0) {
        OP_LOGE(opName, "rankSize is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (rank < 0 || rank >= rankSize) {
        OP_LOGE(opName, "attr rank: %ld is illegal, expected:"\
          "[0 ~ %ld]", rank, rankSize - 1);
        return GRAPH_FAILED;
    }

    int64_t recvCount = 0;
    for (int64_t i = 0; i * i< sendCountMatrix.size(); i++) {
        int64_t tempRecvCount = sendCountMatrix[rank + i * rankSize];
        recvCount += tempRecvCount;
    }

    OP_LOGD(opName, "[%s] alltoallvc recvCount : %ld. rank: %ld rankSize : %ld ", __func__, recvCount, rank, rankSize);
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, recvCount);

    OP_LOGI(opName, "[%s] the op inferShape end.", __func__);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllVCInferDataTypeV2(gert::InferDataTypeContext *context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGI(opName, "[%s] the op inferDataType start.", __func__);

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_LOGI(opName, "[%s] the op inferDataType end.", __func__);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomAllToAllV).InferShape(HcomAllToAllVInferShapeV2).InferDataType(HcomAllToAllVInferDataTypeV2).InputsDataDependency({1,2,3,4});
IMPL_OP_INFERSHAPE(HcomAllToAllVC).InferShape(HcomAllToAllVCInferShapeV2).InferDataType(HcomAllToAllVCInferDataTypeV2).InputsDataDependency({1});
}  // namespace ops