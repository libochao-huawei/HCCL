/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file hcom_ops.cpp
 * \brief
 */
#include "inc/hcom_ops.h"
#include "inc/experiment_ops.h"

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "graph/tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common_shape_fns.h"
#include "op_log.h"
#include "util/util.h"

namespace ge {
// 对外规格约束
constexpr uint32_t MAX_PS_NUM = 256;

// HcomAllGather op
IMPLEMT_INFERFUNC(HcomAllGather, HcomAllGatherInferShape) {
  constexpr int64_t fusionAttrNoFuse = 0;
  constexpr int64_t fusionAttrFuseById = 2;
  constexpr int64_t fusionIdDefaultVal = -1;
  constexpr int64_t fusionIdMinVal = 0;
  constexpr int64_t fusionIdMaxVal = 0x7fffffff;

  int64_t fusionAttr = fusionAttrNoFuse;
  int64_t fusionIdAttr = fusionIdDefaultVal;
  op.GetAttr("fusion", fusionAttr);
  op.GetAttr("fusion_id", fusionIdAttr);
  if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
      OP_LOGE(TbeGetName(op).c_str(), "Attr fusion [%ld] is not supported. expected: [%ld or %ld]",
              fusionAttr, fusionAttrNoFuse, fusionAttrFuseById);
      return GRAPH_FAILED;
  }
  if (fusionAttr == fusionAttrFuseById) {
      if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
          OP_LOGE(TbeGetName(op).c_str(), "In fusion [%ld], attr fusion_id [%ld] is not supported, "\
              "expected: [%ld ~ %ld]", fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
          return GRAPH_FAILED;
      }
  }

  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomAllGather", "Get op name failed.");
    return GRAPH_FAILED;
  }

  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();
  int64_t rankSize = op.get_attr_rank_size();

  auto transformShapeRange = [&]() {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    std::vector<std::pair<int64_t, int64_t>> output_range;
    inTensorDesc.GetShapeRange(input_range);
    int size = input_range.size();
    if (!input_range.empty()) {
        output_range.push_back(ge::GetOutputRange(input_range[0], rankSize));
        for (int i = 1; i < size; ++i) {
            output_range.push_back(input_range[i]);
        }
        OP_LOGI(opName.GetString(), "output_range size: %zu.", output_range.size());
        outTensorDesc.SetShapeRange(output_range);
    }
    op.update_output_desc_y(outTensorDesc);
  };

  if (!ShapeFirstDimDefined(inShape)) {
    outTensorDesc.SetShape(inShape);
    outTensorDesc.SetDataType(inTensorDesc.GetDataType());
    transformShapeRange();
    OP_LOGI(opName.GetString(), "the op infershape end, shape first dim is unknown.");
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> inDims = inShape.GetDims();
  std::vector<int64_t> outDims;
  if (rankSize <= 0) {
    OP_LOGE(opName.GetString(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  outDims = inDims;
  uint32_t fissionFactor = 0;
  if (op.GetAttr("_fission_factor", fissionFactor) == GRAPH_SUCCESS) {
      outDims[0] = inDims[0] * fissionFactor;
  } else {
      outDims[0] = inDims[0] * rankSize;
  }

  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  transformShapeRange();
  OP_LOGI(opName.GetString(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomAllGather, HcomAllGatherVerify) {
  std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  if (rankSize <= 0) {
    OP_LOGE(TbeGetName(op).c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(TbeGetName(op).c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomAllGather, HcomAllGatherVerify);
INFER_FUNC_REG(HcomAllGather, HcomAllGatherInferShape);

// HcomAllGatherV op
IMPLEMT_INFERFUNC(HcomAllGatherV, HcomAllGatherVInferShape) {
  std::vector<string> dep_inputs = {"send_count", "recv_counts", "recv_displacements"};
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  opDesc->SetOpInferDepends(dep_inputs);
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomAllGatherV", "Get op name failed.");
    return GRAPH_FAILED;
  }

  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();

  Tensor recvDispsTensor;
  Tensor recvCountsTensor;
  Tensor sendCountTensor;

  if ((op.GetInputConstData("recv_counts", recvCountsTensor) != GRAPH_SUCCESS) ||
        (op.GetInputConstData("send_count", sendCountTensor) != GRAPH_SUCCESS)) {
        if (inShape.GetDimNum() == 0) {
            OP_LOGE(opName.GetString(), "inShape dim is equal to 0.");
            return GRAPH_FAILED;
        }
        inShape.SetDim(0, -1);
        outTensorDesc.SetShape(inShape);
        outTensorDesc.SetDataType(inTensorDesc.GetDataType());
        op.update_output_desc_y(outTensorDesc);
        OP_LOGI(opName.GetString(), "the op infershape end, shape first dim is unknown.");
        return GRAPH_SUCCESS;
  }

  std::vector<int64_t> inDims = inShape.GetDims();
  std::vector<int64_t> outDims;
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  std::vector<Tensor> vInputVec;
  vector<int64_t> recvDisp;
  vector<int64_t> recvCounts;
  vector<int64_t> sendCount;
  if ((op.GetInputConstData("recv_counts", recvCountsTensor) == GRAPH_SUCCESS) &&
        (op.GetInputConstData("send_count", sendCountTensor) == GRAPH_SUCCESS)) {
    DataType recvCountsDtype = op.GetInputDescByName("recv_counts").GetDataType();
    GetConstValue(op, recvCountsTensor, recvCountsDtype, recvCounts);
    DataType sendCountDtype = op.GetInputDescByName("send_count").GetDataType();
    GetConstValue(op, sendCountTensor, sendCountDtype, sendCount);
    // 计算recvCounts
    for (int i = 0; i < int(recvCounts.size()); i++) {
      for (int j = 1; j < int(inDims.size()); j++) {
          recvCounts[i] *= inDims[j];
      }
    }
    // 计算recvDisp
    int64_t tempsum = 0;
    for (int i = 0; i < int(recvCounts.size()); i++) {
      recvDisp.push_back(tempsum);
      tempsum += recvCounts[i];
    }
    // 计算sendCount
    for (int j = 1; j < int(inDims.size()); j++) {
      sendCount[0] *= inDims[j];
    }
    sendCountTensor.SetData(reinterpret_cast<const uint8_t*>(sendCount.data()), sendCount.size() * sizeof(int64_t));
    recvCountsTensor.SetData(reinterpret_cast<const uint8_t*>(recvCounts.data()), recvCounts.size() * sizeof(int64_t));
    recvDispsTensor.SetData(reinterpret_cast<const uint8_t*>(recvDisp.data()), recvDisp.size() * sizeof(int64_t));
    vInputVec.push_back(recvCountsTensor);
    vInputVec.push_back(recvDispsTensor);
    vInputVec.push_back(sendCountTensor);
    op.SetAttr("vInputVec", vInputVec);
  }
  outDims = inDims;
  int64_t outDim = 0;
  // 计算outdim
  for (int i = 0; i < int(recvCounts.size()); i++) {
      int64_t tempSum = recvDisp[i] + recvCounts[i];
      if (outDim < tempSum) {
          outDim = tempSum;
      }
  }
  for (size_t i = 1; i < inDims.size(); i++) {
      outDim = outDim / inDims[i];
  }
  outDims[0] = outDim;
  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(opName.GetString(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomAllGatherV, HcomAllGatherVVerify) {
  std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
  if (inDims.size() == 0) {
    OP_LOGE(TbeGetName(op).c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomAllGatherV, HcomAllGatherVVerify);
INFER_FUNC_REG(HcomAllGatherV, HcomAllGatherVInferShape);

// HcomReduce op
IMPLEMT_VERIFIER(HcomReduce, HcomReduceVerify) {
  constexpr int64_t fusionAttrNoFuse = 0;
  constexpr int64_t fusionAttrFuseById = 2;
  constexpr int64_t fusionIdDefaultVal = -1;
  constexpr int64_t fusionIdMinVal = 0;
  constexpr int64_t fusionIdMaxVal = 0x7fffffff;
  std::string reduction = op.get_attr_reduction();
  const std::vector<std::string> SUPPORTED_REDUCTION = {"min", "max", "prod", "sum"};
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(TbeGetName(op).c_str(), "Attr reduction [%s] is not supported. expected: min, max, prod, sum",
            reduction.c_str());
    return GRAPH_FAILED;
  }
  int64_t fusionAttr = fusionAttrNoFuse;
  int64_t fusionIdAttr = fusionIdDefaultVal;
  op.GetAttr("fusion", fusionAttr);
  op.GetAttr("fusion_id", fusionIdAttr);
  if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
    OP_LOGE(TbeGetName(op).c_str(), "Attr fusion [%ld] is not supported. expected: [%ld or %ld]",
            fusionAttr, fusionAttrNoFuse, fusionAttrFuseById);
    return GRAPH_FAILED;
  }

  if (fusionAttr == fusionAttrFuseById) {
    if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
      OP_LOGE(TbeGetName(op).c_str(), "In fusion [%ld], attr fusion_id [%ld] is not supported, "\
              "expected: [%ld ~ %ld]", fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomReduce, HcomReduceVerify);
COMMON_INFER_FUNC_REG(HcomReduce, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));

// HcomAllReduce op
IMPLEMT_VERIFIER(HcomAllReduce, HcomAllReduceVerify) {
  constexpr int64_t fusionAttrNoFuse = 0;
  constexpr int64_t fusionAttrFuse = 1;
  constexpr int64_t fusionAttrFuseById = 2;

  constexpr int64_t fusionIdDefaultVal = -1;
  constexpr int64_t fusionIdMinVal = 0;
  constexpr int64_t fusionIdMaxVal = 0x7fffffff;

  std::string reduction = op.get_attr_reduction();
  const std::vector<std::string> SUPPORTED_REDUCTION = {"min", "max", "prod", "sum"};
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(TbeGetName(op).c_str(), "Attr reduction [%s] is not supported. expected: min, max, prod, sum",
            reduction.c_str());
    return GRAPH_FAILED;
  }
  int64_t fusionAttr = fusionAttrFuse;
  int64_t fusionIdAttr = fusionIdDefaultVal;
  op.GetAttr("fusion", fusionAttr);
  op.GetAttr("fusion_id", fusionIdAttr);
  if ((fusionAttr < fusionAttrNoFuse) || (fusionAttr > fusionAttrFuseById)) {
      string fusionValue = std::to_string(fusionAttr);
      ge::ReportPredefinedErrMsg("EI0003", 
      {"ccl_op", "parameter", "value", "tips"},
      {"HcomAllReduce", "fusion", fusionValue.c_str(), "please check fusion setting"});
      OP_LOGE(TbeGetName(op).c_str(), "Attr fusion [%ld] is not supported. expected: [%ld ~ %ld]", fusionAttr,
        fusionAttrNoFuse, fusionAttrFuseById);
    return GRAPH_FAILED;
  }
  if (fusionAttr == fusionAttrFuseById) {
    if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
      OP_LOGE(TbeGetName(op).c_str(), "In fusion [%ld], attr fusion_id [%ld] is not supported, "\
        "expected: [%ld ~ %ld]", fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomAllReduce, HcomAllReduceVerify);
COMMON_INFER_FUNC_REG(HcomAllReduce, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));

// HcomBroadcast op
IMPLEMT_INFERFUNC(HcomBroadcast, HcomBroadcastInferShape) {
  constexpr int64_t fusionAttrNoFuse = 0;
  constexpr int64_t fusionAttrFuseById = 2;
  constexpr int64_t fusionIdDefaultVal = -1;
  constexpr int64_t fusionIdMinVal = 0;
  constexpr int64_t fusionIdMaxVal = 0x7fffffff;

  int64_t fusionAttr = fusionAttrNoFuse;
  int64_t fusionIdAttr = fusionIdDefaultVal;
  op.GetAttr("fusion", fusionAttr);
  op.GetAttr("fusion_id", fusionIdAttr);
  if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
      OP_LOGE(TbeGetName(op).c_str(), "Attr fusion [%ld] is not supported. expected: [%ld or %ld]",
              fusionAttr, fusionAttrNoFuse, fusionAttrFuseById);
      return GRAPH_FAILED;
  }
  if (fusionAttr == fusionAttrFuseById) {
      if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
          OP_LOGE(TbeGetName(op).c_str(), "In fusion [%ld], attr fusion_id [%ld] is not supported, "\
              "expected: [%ld ~ %ld]", fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
          return GRAPH_FAILED;
      }
  }

  const unsigned int UINT_MAX_VALUE = 0xFFFFFFFF;
  auto inputsSize = op.GetInputsSize();
  if (inputsSize >= UINT_MAX_VALUE) {
    OP_LOGE(TbeGetName(op).c_str(), "GetInputsSize [%zu] is more than %u", inputsSize, UINT_MAX_VALUE);
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < inputsSize; i++) {
    auto inputDesc = op.get_dynamic_input_desc_x(i);
    auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
    auto outputDesc = opDesc->MutableOutputDesc(i);
    outputDesc->SetShape(GeShape(inputDesc.GetShape().GetDims()));
    outputDesc->SetDataType(inputDesc.GetDataType());
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomBroadcast, HcomBroadcastVerify) {
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomBroadcast, HcomBroadcastVerify);
INFER_FUNC_REG(HcomBroadcast, HcomBroadcastInferShape);

// HcomReduceScatter op
IMPLEMT_INFERFUNC(HcomReduceScatter, HcomReduceScatterInferShape) {
  constexpr int64_t fusionAttrNoFuse = 0;
  constexpr int64_t fusionAttrFuseById = 2;
  constexpr int64_t fusionIdDefaultVal = -1;
  constexpr int64_t fusionIdMinVal = 0;
  constexpr int64_t fusionIdMaxVal = 0x7fffffff;

  int64_t fusionAttr = fusionAttrNoFuse;
  int64_t fusionIdAttr = fusionIdDefaultVal;
  op.GetAttr("fusion", fusionAttr);
  op.GetAttr("fusion_id", fusionIdAttr);
  if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
      OP_LOGE(TbeGetName(op).c_str(), "Attr fusion [%ld] is not supported. expected: [%ld or %ld]",
              fusionAttr, fusionAttrNoFuse, fusionAttrFuseById);
      return GRAPH_FAILED;
  }
  if (fusionAttr == fusionAttrFuseById) {
      if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
          OP_LOGE(TbeGetName(op).c_str(), "In fusion [%ld], attr fusion_id [%ld] is not supported, "\
              "expected: [%ld ~ %ld]", fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
          return GRAPH_FAILED;
      }
  }
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomReduceScatter", "Get op name failed.");
    return GRAPH_FAILED;
  }

  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();
  if (!ShapeFirstDimDefined(inShape)) {
    outTensorDesc.SetShape(inShape);
    outTensorDesc.SetDataType(inTensorDesc.GetDataType());
    op.update_output_desc_y(outTensorDesc);
    OP_LOGI(opName.GetString(), "the op infershape end, shape first dim is unknown.");
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> inDims = inShape.GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  std::vector<int64_t> outDims;
  if (rankSize <= 0) {
    OP_LOGE(opName.GetString(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  if (inDims[0] % rankSize) {
    OP_LOGE(opName.GetString(),
            "input tensor's first dim is illegal, expected: rankSize[%ld] * N "
            "(N is positive integer), actual: %ld.",
            rankSize, inDims[0]);
    return GRAPH_FAILED;
  }
  outDims = inDims;
  outDims[0] = inDims[0] / rankSize;
  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(opName.GetString(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomReduceScatter, HcomReduceScatterVerify) {
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomReduceScatter", "Get op name failed.");
    return GRAPH_FAILED;
  }

  std::string reduction = op.get_attr_reduction();
  const std::vector<std::string> SUPPORTED_REDUCTION = {"min", "max", "prod", "sum"};
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(opName.GetString(), "Attr reduction [%s] is not supported. expected: min, max, prod, sum",
            reduction.c_str());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  if (rankSize <= 0) {
    OP_LOGE(opName.GetString(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }

  if (ShapeFirstDimDefined(op.get_input_desc_x().GetShape())) {
    if (inDims[0] % rankSize) {
      OP_LOGE(opName.GetString(),
              "input tensor's first dim is illegal, expected: rankSize[%ld] * N "
              "(N is positive integer), actual:%ld.",
              rankSize, inDims[0]);
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(opName.GetString(), "the op verify end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomReduceScatter, HcomReduceScatterInferShape);
VERIFY_FUNC_REG(HcomReduceScatter, HcomReduceScatterVerify);

// HcomReduceScatterV op
IMPLEMT_INFERFUNC(HcomReduceScatterV, HcomReduceScatterVInferShape) {
  std::vector<string> dep_inputs = {"recv_count", "send_counts", "send_displacements"};
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  opDesc->SetOpInferDepends(dep_inputs);

  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomReduceScatterV", "Get op name failed.");
    return GRAPH_FAILED;
  }

  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();

  Tensor recvCountTensor;
  Tensor sendCountsTensor;
  Tensor sendDisplsTensor;

  if ((op.GetInputConstData("recv_count", recvCountTensor) != GRAPH_SUCCESS) ||
        (op.GetInputConstData("send_counts", sendCountsTensor) != GRAPH_SUCCESS)) {
        outTensorDesc.SetShape(inShape);
        outTensorDesc.SetDataType(inTensorDesc.GetDataType());
        op.update_output_desc_y(outTensorDesc);
        OP_LOGI(opName.GetString(), "the op infershape end, shape first dim is unknown.");
        return GRAPH_SUCCESS;
  }

  vector<int64_t> recvCount;
  DataType recvCountDtype = op.GetInputDescByName("recv_count").GetDataType();
  GetConstValue(op, recvCountTensor, recvCountDtype, recvCount);

  vector<int64_t> sendCounts;
  DataType sendCountsDtype = op.GetInputDescByName("send_counts").GetDataType();
  GetConstValue(op, sendCountsTensor, sendCountsDtype, sendCounts);

  vector<int64_t> sendDispls;
  // send displacements为空数组，根据sendCounts构造连续的displacements
  int64_t tmpCount = 0;
  for(size_t i=0; i<sendCounts.size(); i++) {
    sendDispls.push_back(tmpCount);
    tmpCount += sendCounts[i];
  }

  std::vector<int64_t> inDims = inShape.GetDims();
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> outDims;
  outDims = inDims;
  outDims[0] = recvCount[0];

  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  op.update_output_desc_y(outTensorDesc);

  // sendCounts、sendDispls、recvCount乘以其他维的长度
  int64_t otherDims = 1;
  for(size_t i=1; i<inDims.size(); i++) {
    otherDims = otherDims * inDims[i];
  }
  std::transform(recvCount.begin(), recvCount.end(), recvCount.begin(), [otherDims](int num) {
    return num * otherDims;
  });
  std::transform(sendCounts.begin(), sendCounts.end(), sendCounts.begin(), [otherDims](int num) {
    return num * otherDims;
  });
  std::transform(sendDispls.begin(), sendDispls.end(), sendDispls.begin(), [otherDims](int num) {
    return num * otherDims;
  });

  recvCountTensor.SetData(reinterpret_cast<const uint8_t*>(recvCount.data()), recvCount.size() * sizeof(int64_t));
  sendCountsTensor.SetData(reinterpret_cast<const uint8_t*>(sendCounts.data()), sendCounts.size() * sizeof(int64_t));
  sendDisplsTensor.SetData(reinterpret_cast<const uint8_t*>(sendDispls.data()), sendDispls.size() * sizeof(int64_t));

  std::vector<Tensor> vInputVec;
  vInputVec.push_back(recvCountTensor);
  vInputVec.push_back(sendCountsTensor);
  vInputVec.push_back(sendDisplsTensor);
  op.SetAttr("vInputVec", vInputVec);

  OP_LOGI(opName.GetString(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomReduceScatterV, HcomReduceScatterVVerify) {
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomReduceScatterV", "Get op name failed.");
    return GRAPH_FAILED;
  }

  std::string reduction = op.get_attr_reduction();
  const std::vector<std::string> SUPPORTED_REDUCTION = {"min", "max", "sum"};
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(opName.GetString(), "Attr reduction [%s] is not supported. expected: min, max, sum",
            reduction.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }

  OP_LOGI(opName.GetString(), "the op verify end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomReduceScatterV, HcomReduceScatterVInferShape);
VERIFY_FUNC_REG(HcomReduceScatterV, HcomReduceScatterVVerify);

// HcomSend op
IMPLEMT_INFERFUNC(HcomSend, HcomSendInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomSend, HcomSendVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomSend, HcomSendInferShape);
VERIFY_FUNC_REG(HcomSend, HcomSendVerify);

// HcomReceive op
IMPLEMT_INFERFUNC(HcomReceive, HcomReceiveInferShape) {
  TensorDesc outTensorDesc = op.get_output_desc_y();
  std::vector<int64_t> shapeSize{};
  op.GetAttr("shape", shapeSize);
  outTensorDesc.SetShape(ge::Shape(shapeSize));
  uint32_t dataType = op.get_attr_dtype();
  outTensorDesc.SetDataType((DataType)dataType);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(TbeGetName(op).c_str(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomReceive, HcomReceiveVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomReceive, HcomReceiveInferShape);
VERIFY_FUNC_REG(HcomReceive, HcomReceiveVerify);

// HcomAllToAllV op
IMPLEMT_INFERFUNC(HcomAllToAllV, HcomAllToAllVInferShape) {
    std::vector<string> dep_inputs = {"recv_displacements", "recv_counts",
        "send_displacements", "send_counts"};
    auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
    opDesc->SetOpInferDepends(dep_inputs);
    Tensor recvDispTensor;
    Tensor recvCountsTensor;
    if ((op.GetInputConstData("recv_displacements", recvDispTensor) != GRAPH_SUCCESS) ||
        (op.GetInputConstData("recv_counts", recvCountsTensor) != GRAPH_SUCCESS)) {
        auto outTensorDesc = opDesc->MutableOutputDesc("recv_data");
        outTensorDesc->SetShape(GeShape(UNKNOWN_SHAPE));
        outTensorDesc->SetDataType(op.GetInputDescByName("send_data").GetDataType());
        return GRAPH_SUCCESS;
    }

    Tensor sendDispTensor;
    Tensor sendCountsTensor;

    std::vector<Tensor> alltoallvInputVec;
    if ((op.GetInputConstData("send_displacements", sendDispTensor) == GRAPH_SUCCESS) &&
        (op.GetInputConstData("send_counts", sendCountsTensor) == GRAPH_SUCCESS)) {
      alltoallvInputVec.push_back(sendCountsTensor);
      alltoallvInputVec.push_back(sendDispTensor);
      alltoallvInputVec.push_back(recvCountsTensor);
      alltoallvInputVec.push_back(recvDispTensor);
      op.SetAttr("alltoallvInputVec", alltoallvInputVec);
    }

    DataType recvDispDtype = op.GetInputDescByName("recv_displacements").GetDataType();
    vector<int64_t> recvDisp;
    GetConstValue(op, recvDispTensor, recvDispDtype, recvDisp);

    DataType recvCountsDtype = op.GetInputDescByName("recv_counts").GetDataType();
    vector<int64_t> recvCounts;
    GetConstValue(op, recvCountsTensor, recvCountsDtype, recvCounts);

    if (recvDisp.size() != recvCounts.size()) {
        OP_LOGE(TbeGetName(op).c_str(), "recvDisp size[%zu] and recvCounts size[%zu] are different.",
            recvDisp.size(), recvCounts.size());
        return GRAPH_FAILED;
    }

    int64_t recvShape = -1;
    std::vector<int64_t> recvDataDims;
    for (size_t i = 0; i < recvDisp.size(); i++) {
        int64_t tempSum = recvDisp[i] + recvCounts[i];
        if (recvShape < tempSum) {
            recvShape = tempSum;
        }
    }
    recvDataDims.push_back(recvShape);
    auto outTensorDesc = opDesc->MutableOutputDesc("recv_data");
    outTensorDesc->SetShape(GeShape(recvDataDims));
    outTensorDesc->SetDataType(op.GetInputDescByName("send_data").GetDataType());

    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomAllToAllV, HcomAllToAllVVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomAllToAllV, HcomAllToAllVInferShape);
VERIFY_FUNC_REG(HcomAllToAllV, HcomAllToAllVVerify);

// HcomAllToAll op
IMPLEMT_INFERFUNC(HcomAllToAll, HcomAllToAllInferShape) {
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomAllToAll", "Get op name failed.");
    return GRAPH_FAILED;
  }

  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();
  if (!ShapeFullDefined(inShape)) {
    outTensorDesc.SetShape(inShape);
    outTensorDesc.SetDataType(inTensorDesc.GetDataType());
    op.update_output_desc_y(outTensorDesc);
    OP_LOGI(opName.GetString(), "the op infershape end, shape is unknown.");
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> inDims = inShape.GetDims();
  std::vector<int64_t> outDims;
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  outDims = inDims;

  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(opName.GetString(), "the op infershape end");
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomAllToAll, HcomAllToAllVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomAllToAll, HcomAllToAllInferShape);
VERIFY_FUNC_REG(HcomAllToAll, HcomAllToAllVerify);

// HcomAllToAllVC op
IMPLEMT_INFERFUNC(HcomAllToAllVC, HcomAllToAllVCInferShape) {
    constexpr int64_t fusionAttrNoFuse = 0;
    constexpr int64_t fusionAttrFuseById = 2;
    constexpr int64_t fusionIdDefaultVal = -1;
    constexpr int64_t fusionIdMinVal = 0;
    constexpr int64_t fusionIdMaxVal = 0x7fffffff;

    int64_t fusionAttr = fusionAttrNoFuse;
    int64_t fusionIdAttr = fusionIdDefaultVal;
    op.GetAttr("fusion", fusionAttr);
    op.GetAttr("fusion_id", fusionIdAttr);
    if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
        OP_LOGE(TbeGetName(op).c_str(), "Attr fusion [%ld] is not supported. expected: [%ld or %ld]",
                fusionAttr, fusionAttrNoFuse, fusionAttrFuseById);
        return GRAPH_FAILED;
    }
    if (fusionAttr == fusionAttrFuseById) {
        if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
            OP_LOGE(TbeGetName(op).c_str(), "In fusion [%ld], attr fusion_id [%ld] is not supported, "\
                "expected: [%ld ~ %ld]", fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
            return GRAPH_FAILED;
        }
    }

    std::vector<string> dep_inputs = {"send_count_matrix"};
    auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
    opDesc->SetOpInferDepends(dep_inputs);
    Tensor sendCountMatrixTensor;
    if (op.GetInputConstData("send_count_matrix", sendCountMatrixTensor) != GRAPH_SUCCESS) {
        auto outTensorDesc = opDesc->MutableOutputDesc("recv_data");
        outTensorDesc->SetShape(GeShape(UNKNOWN_SHAPE));
        outTensorDesc->SetDataType(op.GetInputDescByName("send_data").GetDataType());
        return GRAPH_SUCCESS;
    }

    DataType sendCountDtype = op.GetInputDescByName("send_count_matrix").GetDataType();
    vector<int64_t> sendCountMatrix;
    GetConstValue(op, sendCountMatrixTensor, sendCountDtype, sendCountMatrix);

    int64_t rank = op.get_attr_rank();
    int64_t rankSize = static_cast<int64_t>(sqrt(sendCountMatrix.size()));
    if (rankSize <= 0) {
        OP_LOGE("HcomAllToAllVC", "rankSize is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (rank < 0 || rank >= rankSize) {
        OP_LOGE("HcomAllToAllVC", "attr rank: %ld is illegal, expected:"\
          "[0 ~ %ld]", rank, rankSize - 1);
        return GRAPH_FAILED;
    }
    int64_t recvCount = 0;
    std::vector<int64_t> recvDataDims;
    for (size_t i = 0; i * i < sendCountMatrix.size(); i++) {
        int64_t tempRecvCount = sendCountMatrix[rank + i * rankSize];
        recvCount += tempRecvCount;
    }
    recvDataDims.push_back(recvCount);
    auto outTensorDesc = opDesc->MutableOutputDesc("recv_data");
    outTensorDesc->SetShape(GeShape(recvDataDims));
    outTensorDesc->SetDataType(op.GetInputDescByName("send_data").GetDataType());

    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomAllToAllVC, HcomAllToAllVCVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomAllToAllVC, HcomAllToAllVCInferShape);
VERIFY_FUNC_REG(HcomAllToAllVC, HcomAllToAllVCVerify);

IMPLEMT_INFERFUNC(HcomRemoteLookup, HcomRemoteLookupInferShape) {
    AscendString opName;
    if (op.GetName(opName) != GRAPH_SUCCESS) {
        OP_LOGE("HcomRemoteLookup", "Get op name failed");
        return GRAPH_FAILED;
    }

    auto outTensorDesc = op.GetOutputDescByName("values");

    std::vector<int64_t> outDims(2, 0);
    std::vector<int64_t> input_shape = op.GetInputDescByName("keys").GetShape().GetDims();

    int maxNum = input_shape[0];
    int embeddingDim = 0;
    op.GetAttr("embedding_dim", embeddingDim);
    outDims[0] = maxNum;
    outDims[1] = embeddingDim;

    ge::Shape outputShape = ge::Shape(outDims);
    ge::DataType outputDtype = outTensorDesc.GetDataType();
    outTensorDesc.SetShape(outputShape);
    outTensorDesc.SetDataType(outputDtype);
    op.UpdateOutputDesc("values", outTensorDesc);
    OP_LOGI(opName.GetString(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomRemoteLookup, HcomRemoteLookupVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomRemoteLookup, HcomRemoteLookupInferShape);
VERIFY_FUNC_REG(HcomRemoteLookup, HcomRemoteLookupVerify);

IMPLEMT_INFERFUNC(HcomCollRemoteLookup, HcomCollRemoteLookupInferShape) {
    AscendString opName;
    if (op.GetName(opName) != GRAPH_SUCCESS) {
        OP_LOGE("HcomCollRemoteLookup", "Get op name failed");
        return GRAPH_FAILED;
    }

    auto outTensorDesc = op.GetOutputDescByName("values");
    std::vector<int64_t> input_shape = op.GetInputDescByName("keys").GetShape().GetDims();
    int maxNum = input_shape[0];

    std::vector<int64_t> outDims(2, 0);
    int embeddingDim = 0;
    if (op.GetAttr("embedding_dim", embeddingDim) != GRAPH_SUCCESS) {
        OP_LOGE("HcomCollRemoteLookup", "Get max_num or embedding_dim failed");
        return GRAPH_FAILED;
    }
    if (maxNum > 0 && embeddingDim > 0) {
        outDims[0] = maxNum;
        outDims[1] = embeddingDim;
    } else {
        OP_LOGE("HcomCollRemoteLookup", "maxNum or embeddingDim invalid value range");
        return GRAPH_FAILED;
    }

    ge::Shape outputShape = ge::Shape(outDims);
    ge::DataType outputDtype = DT_FLOAT; // embedding表数据fp32类型
    outTensorDesc.SetShape(outputShape);
    outTensorDesc.SetDataType(outputDtype);
    op.UpdateOutputDesc("values", outTensorDesc);
    OP_LOGI(opName.GetString(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomCollRemoteLookup, HcomCollRemoteLookupVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomCollRemoteLookup, HcomCollRemoteLookupInferShape);
VERIFY_FUNC_REG(HcomCollRemoteLookup, HcomCollRemoteLookupVerify);

IMPLEMT_INFERFUNC(
    HcomCollRemoteLookupUniquedAndPaired,
    HcomCollRemoteLookupUniquedAndPairedInferShape) {
    AscendString opName;
    if (op.GetName(opName) != GRAPH_SUCCESS) {
        OP_LOGE("HcomCollRemoteLookupUniquedAndPaired", "Get op name failed");
        return GRAPH_FAILED;
    }
    std::vector<int64_t> input_shape = op.GetInputDescByName("keys").GetShape().GetDims();
    int maxNum = input_shape[0];
    int embeddingDim = 0;
    if (op.GetAttr("embedding_dim", embeddingDim) != GRAPH_SUCCESS) {
        OP_LOGE("HcomCollRemoteLookupUniquedAndPaired", "Get max_num or embedding_dim failed");
        return GRAPH_FAILED;
    }
    if ((maxNum <= 0) || (embeddingDim <= 0)) {
        OP_LOGE("HcomCollRemoteLookupUniquedAndPaired", "maxNum or embeddingDim invalid value range");
        return GRAPH_FAILED;
    }
    auto outTensorDescValues = op.GetOutputDescByName("values");
    std::vector<int64_t> outDimsValues(2, 0);
    outDimsValues[0] = maxNum;
    outDimsValues[1] = embeddingDim;
    ge::Shape outputShapeValues = ge::Shape(outDimsValues);
    ge::DataType outputDtypeValues = DT_FLOAT; // embedding表数据fp32类型
    outTensorDescValues.SetShape(outputShapeValues);
    outTensorDescValues.SetDataType(outputDtypeValues);
    op.UpdateOutputDesc("values", outTensorDescValues);

    auto outTensorDescIndices = op.GetOutputDescByName("indices");
    std::vector<int64_t> outDimsIndices(1, maxNum);
    ge::Shape outputShapeIndices = ge::Shape(outDimsIndices);
    ge::DataType outputDtypeIndices = DT_INT64;
    outTensorDescIndices.SetShape(outputShapeIndices);
    outTensorDescIndices.SetDataType(outputDtypeIndices);
    op.UpdateOutputDesc("indices", outTensorDescIndices);

    auto outTensorDescNum = op.GetOutputDescByName("num_uniqued");
    std::vector<int64_t> outDimsNum(1, 1);
    ge::Shape outputShapeNum = ge::Shape(outDimsNum);
    ge::DataType outputDtypeNum = DT_INT64;
    outTensorDescNum.SetShape(outputShapeNum);
    outTensorDescNum.SetDataType(outputDtypeNum);
    op.UpdateOutputDesc("num_uniqued", outTensorDescNum);

    auto outTensorDescPsSeg = op.GetOutputDescByName("ps_segments");
    std::vector<int64_t> outDimsPsSeg(1, MAX_PS_NUM * 2);
    ge::Shape outputShapePsSeg = ge::Shape(outDimsPsSeg);
    ge::DataType outputDtypePsSeg = DT_INT64;
    outTensorDescPsSeg.SetShape(outputShapePsSeg);
    outTensorDescPsSeg.SetDataType(outputDtypePsSeg);
    op.UpdateOutputDesc("ps_segments", outTensorDescPsSeg);

    auto outTensorDescPsSegNum = op.GetOutputDescByName("ps_segments_num");
    std::vector<int64_t> outDimsPsSegNum(1, 1);
    ge::Shape outputShapePsSegNum = ge::Shape(outDimsPsSegNum);
    ge::DataType outputDtypePsSegNum = DT_INT64;
    outTensorDescPsSegNum.SetShape(outputShapePsSegNum);
    outTensorDescPsSegNum.SetDataType(outputDtypePsSegNum);
    op.UpdateOutputDesc("ps_segments_num", outTensorDescPsSegNum);

    OP_LOGI(opName.GetString(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomCollRemoteLookupUniquedAndPaired, HcomCollRemoteLookupUniquedAndPairedVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomCollRemoteLookupUniquedAndPaired, HcomCollRemoteLookupUniquedAndPairedInferShape);
VERIFY_FUNC_REG(HcomCollRemoteLookupUniquedAndPaired, HcomCollRemoteLookupUniquedAndPairedVerify);

IMPLEMT_INFERFUNC(HcomCollRemoteLookupPaired, HcomCollRemoteLookupPairedInferShape) {
    AscendString opName;
    if (op.GetName(opName) != GRAPH_SUCCESS) {
        OP_LOGE("HcomCollRemoteLookupPaired", "Get op name failed");
        return GRAPH_FAILED;
    }
    std::vector<int64_t> input_shape = op.GetInputDescByName("keys").GetShape().GetDims();
    int maxNum = input_shape[0];
    int embeddingDim = 0;
    if (op.GetAttr("embedding_dim", embeddingDim) != GRAPH_SUCCESS) {
        OP_LOGE("HcomCollRemoteLookupPaired", "Get embedding_dim failed");
        return GRAPH_FAILED;
    }
    if ((maxNum <= 0) || (embeddingDim <= 0)) {
        OP_LOGE("HcomCollRemoteLookupPaired", "maxNum or embeddingDim invalid value range");
        return GRAPH_FAILED;
    }
    auto outTensorDescValues = op.GetOutputDescByName("values");
    std::vector<int64_t> outDimsValues(2, 0);

    outDimsValues[0] = maxNum;
    outDimsValues[1] = embeddingDim;
    ge::Shape outputShapeValues = ge::Shape(outDimsValues);
    ge::DataType outputDtypeValues = DT_FLOAT; // embedding表数据fp32类型
    outTensorDescValues.SetShape(outputShapeValues);
    outTensorDescValues.SetDataType(outputDtypeValues);
    op.UpdateOutputDesc("values", outTensorDescValues);

    auto outTensorDescIndices = op.GetOutputDescByName("indices");
    std::vector<int64_t> outDimsIndices(1, maxNum);
    ge::Shape outputShapeIndices = ge::Shape(outDimsIndices);
    ge::DataType outputDtypeIndices = DT_INT64;
    outTensorDescIndices.SetShape(outputShapeIndices);
    outTensorDescIndices.SetDataType(outputDtypeIndices);
    op.UpdateOutputDesc("indices", outTensorDescIndices);

    auto outTensorDescNum = op.GetOutputDescByName("num_uniqued");
    std::vector<int64_t> outDimsNum(1, 1);
    ge::Shape outputShapeNum = ge::Shape(outDimsNum);
    ge::DataType outputDtypeNum = DT_INT64;
    outTensorDescNum.SetShape(outputShapeNum);
    outTensorDescNum.SetDataType(outputDtypeNum);
    op.UpdateOutputDesc("num_uniqued", outTensorDescNum);

    auto outTensorDescPsSeg = op.GetOutputDescByName("ps_segments");
    std::vector<int64_t> outDimsPsSeg(1, MAX_PS_NUM * 2);
    ge::Shape outputShapePsSeg = ge::Shape(outDimsPsSeg);
    ge::DataType outputDtypePsSeg = DT_INT64;
    outTensorDescPsSeg.SetShape(outputShapePsSeg);
    outTensorDescPsSeg.SetDataType(outputDtypePsSeg);
    op.UpdateOutputDesc("ps_segments", outTensorDescPsSeg);

    auto outTensorDescPsSegNum = op.GetOutputDescByName("ps_segments_num");
    std::vector<int64_t> outDimsPsSegNum(1, 1);
    ge::Shape outputShapePsSegNum = ge::Shape(outDimsPsSegNum);
    ge::DataType outputDtypePsSegNum = DT_INT64;
    outTensorDescPsSegNum.SetShape(outputShapePsSegNum);
    outTensorDescPsSegNum.SetDataType(outputDtypePsSegNum);
    op.UpdateOutputDesc("ps_segments_num", outTensorDescPsSegNum);

    OP_LOGI(opName.GetString(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomCollRemoteLookupPaired, HcomCollRemoteLookupPairedVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomCollRemoteLookupPaired, HcomCollRemoteLookupPairedInferShape);
VERIFY_FUNC_REG(HcomCollRemoteLookupPaired, HcomCollRemoteLookupPairedVerify);

IMPLEMT_INFERFUNC(HcomCollRemoteUpdate, HcomCollRemoteUpdateInferShape) {
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomCollRemoteUpdate, HcomCollRemoteUpdateVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomCollRemoteUpdate, HcomCollRemoteUpdateInferShape);
VERIFY_FUNC_REG(HcomCollRemoteUpdate, HcomCollRemoteUpdateVerify);

// HcomCollRemoteUpdatePaired的内容
IMPLEMT_INFERFUNC(HcomCollRemoteUpdatePaired, HcomCollRemoteUpdatePairedInferShape) {
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomCollRemoteUpdatePaired, HcomCollRemoteUpdatePairedVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomCollRemoteUpdatePaired, HcomCollRemoteUpdatePairedInferShape);
VERIFY_FUNC_REG(HcomCollRemoteUpdatePaired, HcomCollRemoteUpdatePairedVerify);

IMPLEMT_INFERFUNC(HcomGather, HcomGatherInferShape) {
    AscendString opName;
    if (op.GetName(opName) != GRAPH_SUCCESS) {
        OP_LOGE("HcomGather", "Get op name failed");
        return GRAPH_FAILED;
    }

    auto inputTensorDesc = op.get_input_desc_x();

    uint32_t outputSize = op.GetOutputsSize();
    const unsigned int UINT_MAX_VALUE = 0xFFFFFFFF;
    if (outputSize >= UINT_MAX_VALUE) {
      OP_LOGE(TbeGetName(op).c_str(), "GetOutputsSize [%u] is more than %u", outputSize, UINT_MAX_VALUE);
      return GRAPH_FAILED;
    }

    auto outTensorDesc = op.get_output_desc_y();
    auto inShape = inputTensorDesc.GetShape();
    if (!ShapeFullDefined(inShape)) {
        outTensorDesc.SetShape(inShape);
        outTensorDesc.SetDataType(inputTensorDesc.GetDataType());
        op.update_output_desc_y(outTensorDesc);
        OP_LOGI(opName.GetString(), "the op HcomGather infershape end, shape is unknown.");
        return GRAPH_SUCCESS;
    }

    std::vector<int64_t> inDims = inShape.GetDims();
    int64_t rankSize = op.get_attr_rank_size();
    std::vector<int64_t> outDims;
    if (rankSize <= 0) {
        OP_LOGE(opName.GetString(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (inDims.size() == 0) {
        OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.",
        inDims.size());
        return GRAPH_FAILED;
    }
    outDims = inDims;
    outDims[0] = inDims[0] * rankSize;
    ge::Shape outputShape = ge::Shape(outDims);
    ge::DataType outputDtype = inputTensorDesc.GetDataType();
    outTensorDesc.SetShape(outputShape);
    outTensorDesc.SetDataType(outputDtype);
    op.update_output_desc_y(outTensorDesc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomGather, HcomGatherVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomGather, HcomGatherInferShape);
VERIFY_FUNC_REG(HcomGather, HcomGatherVerify);
}  // namespace ge
