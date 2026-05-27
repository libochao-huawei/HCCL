/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file op_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_UTIL_H_

#include <memory>
#include <utility>
#include <type_traits>

#include "runtime/tensor.h"

namespace ops {

inline bool IsConstTensor(const gert::Tensor* input_tensor) {
  if (input_tensor != nullptr) {
    if (input_tensor->GetAddr() == nullptr) {
      // empty tensor
      return input_tensor->GetShapeSize() == 0;
    }
    return true;
  }
  return false;
}
}  // namespace ops
#endif  // CANN_OPS_BUILT_IN_OP_UTIL_H_
