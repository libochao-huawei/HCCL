# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 先构建模板层（ring 算法流程模板）。
add_subdirectory(template)

# 当前目录只包含执行器实现源码。
set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/allgather_ring_executor.cc
)

# 把执行器实现编入 hccl 主目标。
target_sources(hccl PRIVATE
    ${src_list}
)
